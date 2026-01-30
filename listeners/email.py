"""
Email Listener - Poll inbox and process incoming emails.

Monitors the agent's email inbox and routes messages to the agent.
Owner emails are auto-replied; external emails let the agent decide.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


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

    # Get email client
    try:
        from agentmail import AgentMail
    except ImportError:
        logger.warning("Email listener disabled: agentmail package not installed")
        return

    api_key = config.get("api_key") or os.environ.get("AGENTMAIL_API_KEY")
    if not api_key:
        logger.warning("Email listener disabled: AGENTMAIL_API_KEY not set")
        return

    client = AgentMail(api_key=api_key)

    # Get inbox ID
    inbox_id = config.get("inbox_id") or os.environ.get("AGENTMAIL_INBOX_ID")
    if not inbox_id:
        try:
            inboxes_response = client.inboxes.list()
            inboxes = getattr(inboxes_response, 'inboxes', None) or inboxes_response
            if inboxes and len(inboxes) > 0:
                inbox = inboxes[0]
                inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
            else:
                inbox = client.inboxes.create()
                inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
        except Exception as e:
            logger.error(f"Email listener disabled: Could not get inbox: {e}")
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
