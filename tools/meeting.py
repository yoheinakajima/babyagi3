"""
Meeting Integration via Recall.ai

Enables the agent to:
1. Join meetings via calendar or email invite
2. Capture real-time transcripts with live backchannel insights
3. Process and summarize meetings post-call
4. Store meeting memories (full transcript, structured extraction, summary)

Architecture:
- Recall.ai handles the actual meeting bot (join, record, transcribe)
- Webhooks deliver real-time and post-meeting data
- This module processes transcripts and integrates with memory

Requires:
- RECALL_API_KEY environment variable
- RECALL_REGION (optional, defaults to us-west-2)
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any

import httpx

from tools import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

RECALL_API_BASE = "https://{region}.recall.ai/api/v1"
DEFAULT_REGION = "us-west-2"


def get_recall_config() -> dict:
    """Get Recall.ai configuration from environment."""
    return {
        "api_key": os.environ.get("RECALL_API_KEY", ""),
        "region": os.environ.get("RECALL_REGION", DEFAULT_REGION),
        "bot_name": os.environ.get("RECALL_BOT_NAME", "BabyAGI Notetaker"),
        "webhook_url": os.environ.get("RECALL_WEBHOOK_URL", ""),  # Your server's public URL
    }


def get_api_base() -> str:
    """Get the Recall.ai API base URL for configured region."""
    config = get_recall_config()
    return RECALL_API_BASE.format(region=config["region"])


# =============================================================================
# Data Models
# =============================================================================

class BotStatus(str, Enum):
    """Recall.ai bot status values."""
    READY = "ready"
    JOINING_CALL = "joining_call"
    IN_WAITING_ROOM = "in_waiting_room"
    IN_CALL_NOT_RECORDING = "in_call_not_recording"
    IN_CALL_RECORDING = "in_call_recording"
    CALL_ENDED = "call_ended"
    DONE = "done"
    FATAL = "fatal"
    ANALYSIS_DONE = "analysis_done"


class MeetingPlatform(str, Enum):
    """Supported meeting platforms."""
    ZOOM = "zoom"
    GOOGLE_MEET = "google_meet"
    MICROSOFT_TEAMS = "microsoft_teams"
    WEBEX = "webex"
    SLACK_HUDDLE = "slack_huddle"
    GOTO_MEETING = "goto_meeting"


@dataclass
class TranscriptWord:
    """A single word in the transcript with timing."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    """A segment of transcript from one speaker."""
    speaker: str
    speaker_id: Optional[str] = None
    text: str = ""
    words: list[TranscriptWord] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    is_final: bool = True


@dataclass
class MeetingParticipant:
    """A participant in the meeting."""
    id: str
    name: str
    is_host: bool = False
    platform_id: Optional[str] = None
    join_time: Optional[datetime] = None
    leave_time: Optional[datetime] = None


@dataclass
class MeetingBot:
    """Represents a Recall.ai meeting bot."""
    id: str
    meeting_url: str
    status: BotStatus = BotStatus.READY
    status_changes: list[dict] = field(default_factory=list)
    join_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    # Meeting metadata
    meeting_title: Optional[str] = None
    platform: Optional[MeetingPlatform] = None

    # Participants
    participants: list[MeetingParticipant] = field(default_factory=list)

    # Transcript data
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)

    # Recording
    video_url: Optional[str] = None
    audio_url: Optional[str] = None


@dataclass
class MeetingFilter:
    """Filter rules for which meetings to auto-join."""
    enabled: bool = True
    min_duration_minutes: int = 0  # Skip meetings shorter than this
    max_duration_minutes: int = 480  # Skip meetings longer than 8 hours
    require_external_attendees: bool = False  # Only join if external people present
    require_invite_keyword: Optional[str] = None  # e.g., "record" in title/description
    exclude_keywords: list[str] = field(default_factory=list)  # Skip if these in title
    calendars: list[str] = field(default_factory=list)  # Specific calendar IDs to monitor
    only_if_invited_email: bool = False  # Only join if bot email is explicitly invited


@dataclass
class MeetingSummary:
    """Structured summary of a meeting."""
    meeting_id: str
    title: str
    date: datetime
    duration_minutes: int
    platform: Optional[str] = None

    # Participants
    attendees: list[dict] = field(default_factory=list)  # [{name, role, speaking_time}]

    # Content
    summary: str = ""  # 2-3 paragraph overview
    key_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)  # [{task, owner, due_date}]
    follow_ups: list[str] = field(default_factory=list)

    # Sentiment/tone
    overall_sentiment: str = "neutral"  # positive, neutral, negative, mixed
    topics_discussed: list[str] = field(default_factory=list)

    # Raw data references
    full_transcript: str = ""
    transcript_word_count: int = 0


# =============================================================================
# Recall.ai API Client
# =============================================================================

class RecallClient:
    """Client for Recall.ai API."""

    def __init__(self, api_key: str = None, region: str = None):
        config = get_recall_config()
        self.api_key = api_key or config["api_key"]
        self.region = region or config["region"]
        self.base_url = RECALL_API_BASE.format(region=self.region)
        self.bot_name = config["bot_name"]
        self.webhook_url = config["webhook_url"]

    def _headers(self) -> dict:
        """Get authorization headers."""
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_bot(
        self,
        meeting_url: str,
        join_at: datetime = None,
        bot_name: str = None,
        recording_mode: str = "speaker_view",
        transcription_provider: str = "recallai_streaming",
    ) -> dict:
        """
        Create a bot to join a meeting.

        Args:
            meeting_url: The meeting URL (Zoom, Meet, Teams, etc.)
            join_at: When to join (None = immediately, or schedule for later)
            bot_name: Name shown in meeting (defaults to configured name)
            recording_mode: "speaker_view", "gallery_view", or "audio_only"
            transcription_provider: "recallai_streaming" or other providers

        Returns:
            Bot object from Recall.ai API
        """
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name or self.bot_name,
            "recording_config": {
                "transcript": {
                    "provider": {
                        transcription_provider: {}
                    }
                },
            },
        }

        # Add scheduled join time if provided
        if join_at:
            # Must be at least 10 minutes in future for scheduled bots
            min_time = datetime.now() + timedelta(minutes=10)
            if join_at > min_time:
                payload["join_at"] = join_at.isoformat()

        # Add real-time webhook endpoints if configured
        if self.webhook_url:
            payload["recording_config"]["realtime_endpoints"] = [
                {
                    "type": "webhook",
                    "url": f"{self.webhook_url}/webhooks/recall/realtime",
                    "events": [
                        "transcript.data",
                        "transcript.partial_data",
                    ]
                }
            ]

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/bot/",
                    headers=self._headers(),
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code in (200, 201):
                    return response.json()
                else:
                    error_text = response.text
                    logger.error(f"Recall API error: {response.status_code} - {error_text}")
                    return {"error": f"API error {response.status_code}: {error_text}"}

            except httpx.TimeoutException:
                return {"error": "Recall API timeout"}
            except Exception as e:
                logger.error(f"Recall API exception: {e}")
                return {"error": str(e)}

    async def get_bot(self, bot_id: str) -> dict:
        """Get bot status and details."""
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/bot/{bot_id}/",
                    headers=self._headers(),
                    timeout=30.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API error {response.status_code}"}

            except Exception as e:
                return {"error": str(e)}

    async def get_bot_transcript(self, bot_id: str) -> dict:
        """Get the transcript for a completed meeting."""
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/bot/{bot_id}/transcript/",
                    headers=self._headers(),
                    timeout=60.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API error {response.status_code}"}

            except Exception as e:
                return {"error": str(e)}

    async def leave_meeting(self, bot_id: str) -> dict:
        """Make the bot leave the meeting."""
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/bot/{bot_id}/leave_call/",
                    headers=self._headers(),
                    timeout=30.0,
                )

                if response.status_code in (200, 204):
                    return {"success": True}
                else:
                    return {"error": f"API error {response.status_code}"}

            except Exception as e:
                return {"error": str(e)}

    async def list_bots(
        self,
        status: str = None,
        created_after: datetime = None,
        limit: int = 50,
    ) -> dict:
        """List bots with optional filters."""
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        params = {"limit": limit}
        if status:
            params["status"] = status
        if created_after:
            params["created_at__gte"] = created_after.isoformat()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/bot/",
                    headers=self._headers(),
                    params=params,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API error {response.status_code}"}

            except Exception as e:
                return {"error": str(e)}

    # =========================================================================
    # Calendar Integration
    # =========================================================================

    async def create_calendar(
        self,
        platform: str,  # "google" or "microsoft"
        oauth_client_id: str,
        oauth_client_secret: str,
    ) -> dict:
        """
        Create a calendar connection.

        This initiates the OAuth flow for connecting a calendar.
        Returns an authorization URL that the user must visit.
        """
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        payload = {
            "platform": platform,
            "oauth_client_id": oauth_client_id,
            "oauth_client_secret": oauth_client_secret,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/calendar/",
                    headers=self._headers(),
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code in (200, 201):
                    return response.json()
                else:
                    return {"error": f"API error {response.status_code}: {response.text}"}

            except Exception as e:
                return {"error": str(e)}

    async def list_calendars(self) -> dict:
        """List connected calendars."""
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/calendar/",
                    headers=self._headers(),
                    timeout=30.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API error {response.status_code}"}

            except Exception as e:
                return {"error": str(e)}

    async def get_calendar_events(
        self,
        calendar_id: str,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> dict:
        """Get events from a connected calendar."""
        if not self.api_key:
            return {"error": "RECALL_API_KEY not configured"}

        params = {}
        if start_time:
            params["start_time__gte"] = start_time.isoformat()
        if end_time:
            params["start_time__lte"] = end_time.isoformat()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/calendar/{calendar_id}/events/",
                    headers=self._headers(),
                    params=params,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API error {response.status_code}"}

            except Exception as e:
                return {"error": str(e)}


# =============================================================================
# Meeting Processor - Handles transcript processing and memory storage
# =============================================================================

class MeetingProcessor:
    """
    Processes meeting transcripts for memory storage.

    Handles:
    1. Real-time transcript accumulation
    2. Live insight generation (backchannel)
    3. Post-meeting summarization
    4. Memory extraction and storage
    """

    def __init__(self, memory=None, senders: dict = None):
        self.memory = memory
        self.senders = senders or {}

        # Active meeting state (bot_id -> state)
        self._active_meetings: dict[str, dict] = {}

        # Backchannel settings
        self.backchannel_enabled = True
        self.backchannel_interval_seconds = 300  # Send insights every 5 minutes
        self.owner_phone = os.environ.get("OWNER_PHONE", "")

        # LLM client for processing
        self._client = None

    @property
    def client(self):
        """Get instrumented Anthropic client."""
        if self._client is None:
            from metrics import InstrumentedAnthropic
            self._client = InstrumentedAnthropic()
        return self._client

    def start_meeting(self, bot_id: str, meeting_url: str, meeting_title: str = None):
        """Initialize state for a new meeting."""
        self._active_meetings[bot_id] = {
            "bot_id": bot_id,
            "meeting_url": meeting_url,
            "meeting_title": meeting_title,
            "started_at": datetime.now(),
            "transcript_segments": [],
            "participants": set(),
            "last_backchannel": datetime.now(),
            "insights_sent": [],
        }
        logger.info(f"Started tracking meeting {bot_id}: {meeting_title or meeting_url}")

    def add_transcript_segment(self, bot_id: str, segment: dict):
        """Add a transcript segment from real-time webhook."""
        if bot_id not in self._active_meetings:
            return

        meeting = self._active_meetings[bot_id]
        meeting["transcript_segments"].append(segment)

        # Track participants
        speaker = segment.get("speaker", "Unknown")
        meeting["participants"].add(speaker)

        # Check if we should send a backchannel insight
        if self.backchannel_enabled:
            time_since_last = datetime.now() - meeting["last_backchannel"]
            if time_since_last.total_seconds() >= self.backchannel_interval_seconds:
                asyncio.create_task(self._send_backchannel_insight(bot_id))

    async def _send_backchannel_insight(self, bot_id: str):
        """Generate and send a backchannel insight via SMS."""
        if bot_id not in self._active_meetings:
            return

        meeting = self._active_meetings[bot_id]
        meeting["last_backchannel"] = datetime.now()

        # Get recent transcript (last 5 minutes worth)
        recent_segments = meeting["transcript_segments"][-50:]  # Last ~50 utterances

        if len(recent_segments) < 5:
            return  # Not enough content yet

        transcript_text = self._format_transcript(recent_segments)

        # Generate insight
        insight = await self._generate_live_insight(
            meeting_title=meeting.get("meeting_title", "Meeting"),
            transcript=transcript_text,
            previous_insights=meeting["insights_sent"],
        )

        if insight and insight.strip():
            meeting["insights_sent"].append(insight)

            # Send via SMS if configured
            if "sendblue" in self.senders and self.owner_phone:
                try:
                    await self.senders["sendblue"].send(
                        to=self.owner_phone,
                        content=f"[Meeting: {meeting.get('meeting_title', 'Live')}]\n\n{insight}",
                    )
                    logger.info(f"Sent backchannel insight for meeting {bot_id}")
                except Exception as e:
                    logger.error(f"Failed to send backchannel: {e}")

    async def _generate_live_insight(
        self,
        meeting_title: str,
        transcript: str,
        previous_insights: list[str],
    ) -> str:
        """Generate a live insight from recent transcript."""

        system_prompt = """You are a helpful meeting assistant providing live backchannel insights.

Your role is to help the meeting participant by surfacing:
- Key points or decisions being made
- Questions they might want to ask
- Important information they might be missing
- Action items being assigned to them
- Suggestions for the discussion

Keep insights:
- Brief (2-3 sentences max)
- Actionable and specific
- Focused on what's most useful RIGHT NOW

If there's nothing particularly noteworthy, respond with just "---" (no insight needed)."""

        previous_context = ""
        if previous_insights:
            previous_context = f"\n\nPrevious insights you've sent:\n" + "\n".join(f"- {i}" for i in previous_insights[-3:])

        user_message = f"""Meeting: {meeting_title}
{previous_context}

Recent transcript:
{transcript}

What's the most useful insight to share right now? (Or "---" if nothing notable)"""

        try:
            from metrics import track_source
            with track_source("meeting_backchannel"):
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

            insight = response.content[0].text.strip()
            if insight == "---" or not insight:
                return ""
            return insight

        except Exception as e:
            logger.error(f"Failed to generate insight: {e}")
            return ""

    def _format_transcript(self, segments: list[dict]) -> str:
        """Format transcript segments into readable text."""
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", seg.get("words", ""))
            if isinstance(text, list):
                text = " ".join(w.get("text", "") for w in text)
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    async def process_completed_meeting(
        self,
        bot_id: str,
        transcript_data: dict,
        meeting_metadata: dict = None,
    ) -> MeetingSummary:
        """
        Process a completed meeting's transcript.

        This is called when the meeting ends and full transcript is available.
        Generates summary, extracts structured data, and stores to memory.
        """
        meeting_metadata = meeting_metadata or {}

        # Parse transcript
        segments = transcript_data.get("results", transcript_data.get("segments", []))
        full_transcript = self._format_transcript(segments)

        # Get participants
        participants = set()
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            participants.add(speaker)

        # Generate structured summary
        summary = await self._generate_meeting_summary(
            title=meeting_metadata.get("title", "Meeting"),
            transcript=full_transcript,
            participants=list(participants),
            duration_minutes=meeting_metadata.get("duration_minutes", 0),
        )

        # Store to memory if available
        if self.memory:
            await self._store_meeting_memory(
                bot_id=bot_id,
                summary=summary,
                full_transcript=full_transcript,
                metadata=meeting_metadata,
            )

        # Clean up active meeting state
        if bot_id in self._active_meetings:
            del self._active_meetings[bot_id]

        return summary

    async def _generate_meeting_summary(
        self,
        title: str,
        transcript: str,
        participants: list[str],
        duration_minutes: int,
    ) -> MeetingSummary:
        """Generate a structured meeting summary using LLM."""

        system_prompt = """You are extracting structured information from a meeting transcript.

You are an AI assistant helping manage information and tasks. Your goal is to capture everything important from this meeting so it can be recalled and acted upon later.

Extract the following:

1. SUMMARY: A 2-3 paragraph overview of what was discussed and accomplished

2. ATTENDEES: List each person with:
   - name: Their name as mentioned
   - role: Their role/title if mentioned (or "Unknown")
   - key_contributions: Brief note on their main contributions

3. KEY_POINTS: The main topics and points discussed (bullet list)

4. DECISIONS: Specific decisions that were made (bullet list)

5. ACTION_ITEMS: Tasks assigned during the meeting:
   - task: What needs to be done
   - owner: Who is responsible (or "Unassigned")
   - due_date: When it's due (or null)
   - priority: high/medium/low

6. FOLLOW_UPS: Next steps, future meetings, things to revisit

7. TOPICS: Main themes/subjects discussed (for categorization)

8. SENTIMENT: Overall tone (positive/neutral/negative/mixed)

Return valid JSON matching this schema:
{
    "summary": "string",
    "attendees": [{"name": "string", "role": "string", "key_contributions": "string"}],
    "key_points": ["string"],
    "decisions": ["string"],
    "action_items": [{"task": "string", "owner": "string", "due_date": "string|null", "priority": "string"}],
    "follow_ups": ["string"],
    "topics": ["string"],
    "sentiment": "string"
}"""

        # Truncate very long transcripts
        max_chars = 100000  # ~25k tokens
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "\n\n[... transcript truncated for processing ...]"

        user_message = f"""Meeting: {title}
Duration: {duration_minutes} minutes
Participants: {', '.join(participants)}

TRANSCRIPT:
{transcript}

Extract structured information from this meeting."""

        try:
            from metrics import track_source
            with track_source("meeting_summary"):
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

            response_text = response.content[0].text

            # Parse JSON
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response_text[start:end])
            else:
                data = {}

            return MeetingSummary(
                meeting_id="",  # Will be set by caller
                title=title,
                date=datetime.now(),
                duration_minutes=duration_minutes,
                attendees=data.get("attendees", []),
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                decisions=data.get("decisions", []),
                action_items=data.get("action_items", []),
                follow_ups=data.get("follow_ups", []),
                overall_sentiment=data.get("sentiment", "neutral"),
                topics_discussed=data.get("topics", []),
                full_transcript=transcript,
                transcript_word_count=len(transcript.split()),
            )

        except Exception as e:
            logger.error(f"Failed to generate meeting summary: {e}")
            return MeetingSummary(
                meeting_id="",
                title=title,
                date=datetime.now(),
                duration_minutes=duration_minutes,
                summary=f"Meeting summary generation failed: {e}",
                full_transcript=transcript,
                transcript_word_count=len(transcript.split()),
            )

    async def _store_meeting_memory(
        self,
        bot_id: str,
        summary: MeetingSummary,
        full_transcript: str,
        metadata: dict,
    ):
        """Store meeting data to the memory system."""
        if not self.memory:
            return

        # 1. Store the full transcript as an event
        transcript_event = self.memory.log_event(
            content=f"[Meeting Transcript: {summary.title}]\n\n{full_transcript}",
            event_type="meeting_transcript",
            channel="meeting",
            direction="inbound",
            is_owner=True,
            metadata={
                "bot_id": bot_id,
                "meeting_title": summary.title,
                "duration_minutes": summary.duration_minutes,
                "word_count": summary.transcript_word_count,
                "platform": metadata.get("platform"),
            },
        )

        # 2. Store the structured summary as an event
        summary_content = f"""[Meeting Summary: {summary.title}]
Date: {summary.date.isoformat()}
Duration: {summary.duration_minutes} minutes
Attendees: {', '.join(a.get('name', 'Unknown') for a in summary.attendees)}

SUMMARY:
{summary.summary}

KEY POINTS:
{chr(10).join('- ' + p for p in summary.key_points)}

DECISIONS:
{chr(10).join('- ' + d for d in summary.decisions)}

ACTION ITEMS:
{chr(10).join('- ' + a.get('task', '') + ' (Owner: ' + a.get('owner', 'Unassigned') + ')' for a in summary.action_items)}

FOLLOW-UPS:
{chr(10).join('- ' + f for f in summary.follow_ups)}

Topics: {', '.join(summary.topics_discussed)}
Sentiment: {summary.overall_sentiment}"""

        summary_event = self.memory.log_event(
            content=summary_content,
            event_type="meeting_summary",
            channel="meeting",
            direction="internal",
            is_owner=True,
            parent_event_id=transcript_event.id,
            metadata={
                "bot_id": bot_id,
                "meeting_title": summary.title,
                "attendees": summary.attendees,
                "action_items": summary.action_items,
                "decisions": summary.decisions,
            },
        )

        logger.info(f"Stored meeting memory: transcript={transcript_event.id}, summary={summary_event.id}")


# =============================================================================
# Active Meetings Manager - Singleton for tracking ongoing meetings
# =============================================================================

_meeting_processor: Optional[MeetingProcessor] = None


def get_meeting_processor() -> MeetingProcessor:
    """Get the global meeting processor instance."""
    global _meeting_processor
    if _meeting_processor is None:
        _meeting_processor = MeetingProcessor()
    return _meeting_processor


def initialize_meeting_processor(memory=None, senders: dict = None):
    """Initialize the meeting processor with memory and senders."""
    global _meeting_processor
    _meeting_processor = MeetingProcessor(memory=memory, senders=senders)
    return _meeting_processor


# =============================================================================
# Agent Tools
# =============================================================================

@tool
async def join_meeting(
    meeting_url: str,
    join_at: str = None,
    bot_name: str = None,
) -> dict:
    """
    Send a bot to join a video meeting and take notes.

    Supports: Zoom, Google Meet, Microsoft Teams, Webex, Slack Huddles, GoTo Meeting.

    Args:
        meeting_url: The meeting URL to join
        join_at: Optional ISO datetime to schedule join (must be 10+ min in future)
        bot_name: Optional custom name for the bot in the meeting

    Returns:
        Bot details including ID for tracking

    Example:
        join_meeting(meeting_url="https://zoom.us/j/123456789")
        join_meeting(meeting_url="https://meet.google.com/abc-defg-hij", join_at="2024-01-15T14:00:00")
    """
    client = RecallClient()

    # Parse join_at if provided
    scheduled_time = None
    if join_at:
        try:
            scheduled_time = datetime.fromisoformat(join_at.replace("Z", "+00:00"))
        except ValueError:
            return {"error": f"Invalid datetime format: {join_at}. Use ISO format."}

    result = await client.create_bot(
        meeting_url=meeting_url,
        join_at=scheduled_time,
        bot_name=bot_name,
    )

    if "error" not in result:
        # Initialize tracking for this meeting
        processor = get_meeting_processor()
        processor.start_meeting(
            bot_id=result.get("id", ""),
            meeting_url=meeting_url,
            meeting_title=result.get("meeting_metadata", {}).get("title"),
        )

    return result


@tool
async def get_meeting_status(bot_id: str) -> dict:
    """
    Get the status of a meeting bot.

    Args:
        bot_id: The bot ID returned from join_meeting

    Returns:
        Current status and meeting details
    """
    client = RecallClient()
    return await client.get_bot(bot_id)


@tool
async def leave_meeting(bot_id: str) -> dict:
    """
    Make a bot leave a meeting early.

    Args:
        bot_id: The bot ID to remove from meeting

    Returns:
        Success status
    """
    client = RecallClient()
    return await client.leave_meeting(bot_id)


@tool
async def get_meeting_transcript(bot_id: str) -> dict:
    """
    Get the transcript from a completed meeting.

    Args:
        bot_id: The bot ID for the meeting

    Returns:
        Full transcript with speaker attribution
    """
    client = RecallClient()
    return await client.get_bot_transcript(bot_id)


@tool
async def list_meeting_bots(
    status: str = None,
    days_back: int = 7,
) -> dict:
    """
    List recent meeting bots.

    Args:
        status: Filter by status (e.g., "done", "in_call_recording")
        days_back: How many days back to search (default 7)

    Returns:
        List of bots with their status
    """
    client = RecallClient()
    created_after = datetime.now() - timedelta(days=days_back)
    return await client.list_bots(status=status, created_after=created_after)


@tool
async def list_connected_calendars() -> dict:
    """
    List calendars connected to the meeting bot system.

    Returns:
        List of connected calendars with their status
    """
    client = RecallClient()
    return await client.list_calendars()


@tool
async def search_meeting_memories(
    query: str,
    limit: int = 10,
) -> dict:
    """
    Search through past meeting memories.

    Args:
        query: Search query (e.g., "action items from product meeting", "decisions about pricing")
        limit: Maximum results to return

    Returns:
        Matching meeting summaries and transcripts
    """
    processor = get_meeting_processor()
    if not processor.memory:
        return {"error": "Memory system not initialized"}

    # Search for meeting events
    events = processor.memory.search_events(query, limit=limit * 2)

    # Filter to meeting events
    meeting_events = [
        e for e in events
        if e.event_type in ("meeting_transcript", "meeting_summary")
    ][:limit]

    return {
        "meetings": [
            {
                "id": e.id,
                "type": e.event_type,
                "title": e.metadata.get("meeting_title", "Unknown") if e.metadata else "Unknown",
                "date": e.timestamp.isoformat() if e.timestamp else None,
                "content_preview": e.content[:500] + "..." if len(e.content) > 500 else e.content,
            }
            for e in meeting_events
        ]
    }


# =============================================================================
# Email Invite Parser - For handling forwarded meeting invites
# =============================================================================

def parse_meeting_url_from_email(email_content: str) -> Optional[str]:
    """
    Extract a meeting URL from an email (e.g., forwarded calendar invite).

    Looks for URLs from supported platforms:
    - Zoom: zoom.us/j/...
    - Google Meet: meet.google.com/...
    - Microsoft Teams: teams.microsoft.com/...
    - Webex: *.webex.com/...
    """
    # Patterns for meeting URLs
    patterns = [
        r'https?://[^\s]*zoom\.us/j/[^\s<>"\']+',
        r'https?://meet\.google\.com/[a-z-]+',
        r'https?://teams\.microsoft\.com/[^\s<>"\']+',
        r'https?://[^\s]*\.webex\.com/[^\s<>"\']+',
        r'https?://[^\s]*gotomeeting\.com/[^\s<>"\']+',
        r'https?://app\.slack\.com/huddle/[^\s<>"\']+',
    ]

    for pattern in patterns:
        match = re.search(pattern, email_content, re.IGNORECASE)
        if match:
            url = match.group(0)
            # Clean up trailing punctuation
            url = url.rstrip('.,;:!?"\'>)')
            return url

    return None


def parse_meeting_time_from_email(email_content: str) -> Optional[datetime]:
    """
    Extract meeting time from an email (calendar invite).

    Looks for common date/time patterns in calendar invites.
    """
    # This is a simplified parser - in production, you'd use icalendar library
    # to properly parse .ics attachments

    # Look for DTSTART in ICS format
    dtstart_match = re.search(r'DTSTART[^:]*:(\d{8}T\d{6}Z?)', email_content)
    if dtstart_match:
        dt_str = dtstart_match.group(1)
        try:
            if dt_str.endswith('Z'):
                return datetime.strptime(dt_str, "%Y%m%dT%H%M%SZ")
            else:
                return datetime.strptime(dt_str, "%Y%m%dT%H%M%S")
        except ValueError:
            pass

    return None


@tool(env=["RECALL_API_KEY"])
async def connect_calendar(
    platform: str,
    oauth_client_id: str = None,
    oauth_client_secret: str = None,
) -> dict:
    """
    Connect a calendar for automatic meeting bot scheduling.

    This initiates an OAuth flow to connect Google or Microsoft calendar.
    Once connected, the system can automatically join meetings based on your settings.

    Args:
        platform: "google" or "microsoft"
        oauth_client_id: OAuth client ID (or set GOOGLE_OAUTH_CLIENT_ID / MICROSOFT_OAUTH_CLIENT_ID env var)
        oauth_client_secret: OAuth client secret (or set GOOGLE_OAUTH_CLIENT_SECRET / MICROSOFT_OAUTH_CLIENT_SECRET env var)

    Returns:
        Authorization URL to complete OAuth flow, or error

    Example:
        connect_calendar(platform="google")
    """
    platform = platform.lower()
    if platform not in ("google", "microsoft"):
        return {"error": f"Unsupported platform: {platform}. Use 'google' or 'microsoft'."}

    # Get credentials from env if not provided
    if not oauth_client_id:
        env_var = f"{platform.upper()}_OAUTH_CLIENT_ID"
        oauth_client_id = os.environ.get(env_var)
        if not oauth_client_id:
            return {
                "error": f"OAuth client ID required. Either pass it directly or set {env_var} env var.",
                "setup_instructions": f"""To connect {platform.title()} Calendar:
1. Go to {
    'https://console.cloud.google.com/apis/credentials' if platform == 'google' else
    'https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps'
}
2. Create an OAuth 2.0 client
3. Set the redirect URI to your server's callback URL
4. Set the {env_var} and {platform.upper()}_OAUTH_CLIENT_SECRET env vars
5. Run this command again"""
            }

    if not oauth_client_secret:
        env_var = f"{platform.upper()}_OAUTH_CLIENT_SECRET"
        oauth_client_secret = os.environ.get(env_var)
        if not oauth_client_secret:
            return {"error": f"OAuth client secret required. Set {env_var} env var."}

    client = RecallClient()
    result = await client.create_calendar(
        platform=platform,
        oauth_client_id=oauth_client_id,
        oauth_client_secret=oauth_client_secret,
    )

    if "error" not in result:
        auth_url = result.get("oauth_authorize_url", result.get("authorize_url"))
        if auth_url:
            return {
                "status": "oauth_required",
                "message": f"Please visit this URL to authorize calendar access:",
                "authorization_url": auth_url,
                "calendar_id": result.get("id"),
            }

    return result


@tool(env=["RECALL_API_KEY"])
async def get_calendar_events(
    calendar_id: str = None,
    days_ahead: int = 7,
) -> dict:
    """
    Get upcoming events from a connected calendar.

    Args:
        calendar_id: The calendar ID (from connect_calendar). If not provided, uses first connected calendar.
        days_ahead: How many days ahead to look (default 7)

    Returns:
        List of upcoming events with meeting links
    """
    client = RecallClient()

    # If no calendar_id provided, list calendars and use first one
    if not calendar_id:
        calendars = await client.list_calendars()
        if "error" in calendars:
            return calendars
        cal_list = calendars.get("results", [])
        if not cal_list:
            return {"error": "No calendars connected. Use connect_calendar first."}
        calendar_id = cal_list[0].get("id")

    # Get events
    start_time = datetime.now()
    end_time = start_time + timedelta(days=days_ahead)

    result = await client.get_calendar_events(
        calendar_id=calendar_id,
        start_time=start_time,
        end_time=end_time,
    )

    if "error" in result:
        return result

    # Format events for readability
    events = result.get("results", [])
    return {
        "calendar_id": calendar_id,
        "events": [
            {
                "title": e.get("summary", "No title"),
                "start_time": e.get("start_time"),
                "end_time": e.get("end_time"),
                "meeting_url": e.get("meeting_url"),
                "has_bot_scheduled": e.get("bot_id") is not None,
            }
            for e in events
        ]
    }


@tool
async def configure_meeting_bot(
    setting: str,
    value: str = None,
) -> dict:
    """
    Configure meeting bot settings.

    Args:
        setting: The setting to configure:
            - "bot_name": Name shown in meetings (e.g., "MyCompany Notetaker")
            - "backchannel_interval": Minutes between live insights (e.g., "5")
            - "backchannel_enabled": Enable/disable live SMS insights ("true"/"false")
            - "show_settings": Display current settings
        value: The value to set (not needed for show_settings)

    Returns:
        Updated settings or current configuration
    """
    processor = get_meeting_processor()
    config = get_recall_config()

    if setting == "show_settings":
        return {
            "bot_name": config.get("bot_name", "BabyAGI Notetaker"),
            "region": config.get("region", "us-west-2"),
            "backchannel_enabled": processor.backchannel_enabled,
            "backchannel_interval_seconds": processor.backchannel_interval_seconds,
            "webhook_url": config.get("webhook_url", "Not configured"),
            "api_key_configured": bool(config.get("api_key")),
        }

    if setting == "bot_name" and value:
        # Note: This only affects new bots, requires env var change for persistence
        return {
            "note": "To change bot name persistently, set RECALL_BOT_NAME env var",
            "current_value": config.get("bot_name"),
            "requested_value": value,
        }

    if setting == "backchannel_interval" and value:
        try:
            minutes = int(value)
            processor.backchannel_interval_seconds = minutes * 60
            return {"backchannel_interval_minutes": minutes}
        except ValueError:
            return {"error": f"Invalid interval: {value}. Use a number of minutes."}

    if setting == "backchannel_enabled" and value:
        enabled = value.lower() in ("true", "yes", "1", "on")
        processor.backchannel_enabled = enabled
        return {"backchannel_enabled": enabled}

    return {"error": f"Unknown setting: {setting}. Use 'show_settings' to see available options."}


async def handle_meeting_invite_email(
    email_content: str,
    email_subject: str = "",
    auto_join: bool = True,
) -> dict:
    """
    Handle a forwarded meeting invite email.

    Extracts meeting URL and time, optionally schedules bot to join.

    Args:
        email_content: The email body (may include .ics attachment content)
        email_subject: The email subject line
        auto_join: Whether to automatically schedule the bot

    Returns:
        Parsed meeting info and bot scheduling result
    """
    meeting_url = parse_meeting_url_from_email(email_content)
    meeting_time = parse_meeting_time_from_email(email_content)

    if not meeting_url:
        return {
            "error": "No meeting URL found in email",
            "hint": "Forward a calendar invite containing a Zoom, Meet, Teams, or Webex link",
        }

    result = {
        "meeting_url": meeting_url,
        "meeting_time": meeting_time.isoformat() if meeting_time else None,
        "subject": email_subject,
    }

    if auto_join:
        # Schedule bot to join
        join_result = await join_meeting(
            meeting_url=meeting_url,
            join_at=meeting_time.isoformat() if meeting_time else None,
        )
        result["bot"] = join_result

        if "error" not in join_result:
            result["status"] = "Bot scheduled to join meeting"
        else:
            result["status"] = f"Failed to schedule bot: {join_result.get('error')}"

    return result
