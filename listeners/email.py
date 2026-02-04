"""
Email Listener - Poll inbox and process incoming emails.

Monitors the agent's email inbox and routes messages to the agent.
Owner emails are auto-replied; external emails let the agent decide.

Also detects meeting invites and automatically schedules bots to join.
"""

import asyncio
import logging
import os
import re

from utils.email_client import get_client, get_inbox_id

logger = logging.getLogger(__name__)


# =============================================================================
# Meeting Invite Detection
# =============================================================================

def is_meeting_invite(subject: str, body: str) -> bool:
    """
    Detect if an email is likely a meeting invite.

    Checks for:
    - Calendar invite patterns (.ics, VCALENDAR)
    - Meeting URLs (Zoom, Meet, Teams, etc.)
    - Common invitation phrases
    """
    # Check for ICS/calendar content
    if "BEGIN:VCALENDAR" in body or "text/calendar" in body.lower():
        return True

    # Check for meeting URLs
    meeting_url_patterns = [
        r'zoom\.us/j/',
        r'meet\.google\.com/',
        r'teams\.microsoft\.com/',
        r'webex\.com/',
        r'gotomeeting\.com/',
    ]
    for pattern in meeting_url_patterns:
        if re.search(pattern, body, re.IGNORECASE):
            return True

    # Check subject for invite keywords
    invite_keywords = [
        "invitation:", "invite:", "meeting invite",
        "calendar invite", "you're invited", "join meeting",
    ]
    subject_lower = subject.lower()
    for keyword in invite_keywords:
        if keyword in subject_lower:
            return True

    return False


async def handle_meeting_invite(
    email_body: str,
    email_subject: str,
    sender: str,
    owner_phone: str = None,
    sendblue_sender = None,
) -> dict:
    """
    Handle a detected meeting invite email.

    Extracts meeting URL and schedules a bot to join.
    Optionally notifies owner via SMS.

    Returns:
        Result dict with meeting info and bot status
    """
    try:
        from tools.meeting import handle_meeting_invite_email

        result = await handle_meeting_invite_email(
            email_content=email_body,
            email_subject=email_subject,
            auto_join=True,
        )

        # Notify owner if SMS is configured
        if owner_phone and sendblue_sender and "error" not in result:
            bot_info = result.get("bot", {})
            bot_id = bot_info.get("id", "unknown")
            meeting_time = result.get("meeting_time", "unknown time")

            notification = (
                f"Meeting invite received from {sender}\n"
                f"Subject: {email_subject}\n"
                f"Time: {meeting_time}\n"
                f"Bot scheduled: {bot_id[:8]}..."
            )

            try:
                await sendblue_sender.send(to=owner_phone, content=notification)
            except Exception as e:
                logger.error(f"Failed to send meeting notification: {e}")

        return result

    except ImportError:
        logger.warning("Meeting tools not available, skipping invite handling")
        return {"error": "Meeting tools not available"}
    except Exception as e:
        logger.error(f"Error handling meeting invite: {e}")
        return {"error": str(e)}


async def run_email_listener(agent, config: dict = None):
    """Run the email listener.

    Args:
        agent: The Agent instance
        config: Configuration dict with:
            - owner_email: Owner's email address (for owner detection)
            - poll_interval: Seconds between inbox checks (default: 60)
            - api_key: AgentMail API key (or use AGENTMAIL_API_KEY env)
            - inbox_id: Inbox to monitor (or use AGENTMAIL_INBOX_ID env)
    """
    config = config or {}
    owner_email = config.get("owner_email") or os.environ.get("OWNER_EMAIL")
    poll_interval = config.get("poll_interval", 60)

    # Get email client using shared singleton
    client = get_client()
    if not client:
        logger.warning("Email listener disabled: agentmail not available or AGENTMAIL_API_KEY not set")
        return

    # Get inbox ID using shared logic
    inbox_id = config.get("inbox_id") or get_inbox_id()
    if not inbox_id:
        logger.error("Email listener disabled: Could not get inbox")
        return

    logger.info(f"Email listener started for {inbox_id}")
    processed_ids = set()

    while True:
        try:
            # Fetch messages
            messages_response = client.inboxes.messages.list(inbox_id=inbox_id)
            messages = getattr(messages_response, 'messages', None) or messages_response or []

            for msg in messages:
                msg_id = getattr(msg, 'message_id', None) or getattr(msg, 'id', None)

                # Skip already processed
                if msg_id in processed_ids:
                    continue

                # Skip read messages (already processed in previous runs)
                if getattr(msg, 'read', False):
                    processed_ids.add(msg_id)
                    continue

                # Extract message details
                sender = getattr(msg, 'from_', '') or getattr(msg, 'sender', '') or ''
                subject = getattr(msg, 'subject', '') or ''

                # Read full message
                try:
                    full_msg = client.inboxes.messages.get(inbox_id=inbox_id, message_id=msg_id)
                    body = getattr(full_msg, 'body', None) or getattr(full_msg, 'text', '') or ''
                except Exception:
                    body = getattr(msg, 'snippet', '') or ''

                # Determine if owner
                is_owner = bool(owner_email and sender.lower() == owner_email.lower())

                # Build thread ID
                if is_owner:
                    # Owner emails use message thread for context
                    thread_id = f"email:{msg_id}"
                else:
                    # External emails get separate thread per sender
                    thread_id = f"external:{sender}"

                # Build context
                context = {
                    "channel": "email",
                    "is_owner": is_owner,
                    "sender": sender,
                    "subject": subject,
                    "message_id": msg_id,
                    "reply_to": msg_id,
                }

                # Check if this is a meeting invite
                if is_meeting_invite(subject, body):
                    logger.info(f"Detected meeting invite from {sender}: {subject[:50]}")

                    # Get SMS sender for notifications
                    sendblue_sender = agent.senders.get("sendblue")
                    owner_phone = os.environ.get("OWNER_PHONE", "")

                    # Handle the meeting invite
                    invite_result = await handle_meeting_invite(
                        email_body=body,
                        email_subject=subject,
                        sender=sender,
                        owner_phone=owner_phone if is_owner else None,
                        sendblue_sender=sendblue_sender,
                    )

                    if "error" not in invite_result:
                        logger.info(f"Meeting bot scheduled for invite: {invite_result.get('meeting_url')}")
                        # Mark as processed and continue (don't also send to agent)
                        processed_ids.add(msg_id)
                        continue
                    else:
                        logger.warning(f"Meeting invite handling failed: {invite_result.get('error')}")
                        # Fall through to normal processing

                # Format input with email context
                email_input = f"[Email from {sender}]\nSubject: {subject}\n\n{body}"

                logger.info(f"Processing email from {sender}: {subject[:50]}")

                # Process through agent
                response = await agent.run_async(
                    email_input,
                    thread_id=thread_id,
                    context=context
                )

                # Auto-reply for owner emails
                if is_owner and response:
                    try:
                        # Use the email sender if registered
                        if "email" in agent.senders:
                            await agent.senders["email"].send(
                                to=sender,
                                content=response,
                                subject=f"Re: {subject}"
                            )
                        else:
                            # Fall back to direct client
                            client.inboxes.messages.send(
                                inbox_id=inbox_id,
                                to=sender,
                                subject=f"Re: {subject}",
                                text=response
                            )
                        logger.info(f"Replied to {sender}")
                    except Exception as e:
                        logger.error(f"Failed to reply: {e}")

                # Mark as processed
                processed_ids.add(msg_id)

        except Exception as e:
            logger.error(f"Email listener error: {e}")

        await asyncio.sleep(poll_interval)
