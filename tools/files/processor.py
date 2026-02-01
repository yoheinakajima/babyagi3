"""
Document Processor

Processes files using LLM to extract facts, entities, and determine organization.
Uses a separate LLM call (outside main agent loop) to keep context clean.
"""

import base64
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

from anthropic import Anthropic

# Cost per token (approximate for claude-sonnet-4-20250514)
INPUT_COST_PER_TOKEN = 0.003 / 1000  # $3 per million tokens
OUTPUT_COST_PER_TOKEN = 0.015 / 1000  # $15 per million tokens


class DocumentProcessor:
    """
    Processes documents using LLM to extract structured knowledge.

    This runs OUTSIDE the main agent loop to:
    1. Keep main context clean (full docs never enter main thread)
    2. Enable parallel processing
    3. Allow flexible extraction without context bloat
    """

    def __init__(self, memory_store=None):
        self.client = Anthropic()
        self.store = memory_store
        self.model = "claude-sonnet-4-20250514"

    def process_file(
        self,
        file_path: str,
        user_context: dict | None = None,
        existing_projects: list[str] | None = None,
    ) -> dict:
        """
        Process a file and extract structured knowledge.

        Args:
            file_path: Path to the file
            user_context: Context about user (preferences, objectives, etc.)
            existing_projects: List of existing project folder names

        Returns:
            {
                "summary": "Brief summary of the document",
                "facts": [
                    {
                        "subject": "entity name",
                        "predicate": "verb/relationship",
                        "object": "entity name or value",
                        "object_type": "entity|value|text",
                        "fact_type": "relation|attribute|event|state|metric",
                        "fact_text": "Full natural sentence",
                        "confidence": 0.0-1.0,
                        "valid_from": "ISO date or null",
                        "valid_to": "ISO date or null",
                        "mentioned_entities": ["other entities in context"]
                    }
                ],
                "entities": [
                    {
                        "name": "...",
                        "type": "person|org|tool|concept|location|file",
                        "description": "...",
                        "aliases": [...]
                    }
                ],
                "suggested_project": "project folder name",
                "tags": ["tag1", "tag2"],
                "extraction_call": {
                    "model": "...",
                    "input_tokens": ...,
                    "output_tokens": ...,
                    "cost_usd": ...,
                    "duration_ms": ...
                }
            }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and read content
        mime_type = self._get_mime_type(path)
        content, is_image = self._read_file_content(path, mime_type)

        # Build context
        context = self._build_context(user_context, existing_projects)

        # Call LLM for extraction
        start_time = time.time()
        result = self._extract_with_llm(content, is_image, mime_type, path.name, context)
        duration_ms = int((time.time() - start_time) * 1000)

        # Add timing info
        result["extraction_call"]["duration_ms"] = duration_ms

        return result

    def process_content(
        self,
        content: str | bytes,
        filename: str,
        mime_type: str | None = None,
        user_context: dict | None = None,
        existing_projects: list[str] | None = None,
    ) -> dict:
        """
        Process content directly (for attachments, etc.).

        Same return format as process_file.
        """
        # Determine if this is an image
        is_image = mime_type and mime_type.startswith("image/")

        # Build context
        context = self._build_context(user_context, existing_projects)

        # Call LLM for extraction
        start_time = time.time()
        result = self._extract_with_llm(content, is_image, mime_type, filename, context)
        duration_ms = int((time.time() - start_time) * 1000)

        result["extraction_call"]["duration_ms"] = duration_ms

        return result

    def _get_mime_type(self, path: Path) -> str:
        """Detect mime type from file extension."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))

        # Handle common types
        ext = path.suffix.lower()
        if ext in (".pdf",):
            return "application/pdf"
        elif ext in (".docx",):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext in (".doc",):
            return "application/msword"
        elif ext in (".csv",):
            return "text/csv"
        elif ext in (".xlsx", ".xls"):
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            return mime_type or f"image/{ext[1:]}"

        return mime_type or "application/octet-stream"

    def _read_file_content(self, path: Path, mime_type: str) -> tuple[str | bytes, bool]:
        """Read file content, extracting text where possible."""
        is_image = mime_type.startswith("image/")

        if is_image:
            # Return raw bytes for images
            return path.read_bytes(), True

        # Try to extract text from various formats
        try:
            if mime_type == "application/pdf":
                return self._extract_pdf_text(path), False
            elif mime_type in (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ):
                return self._extract_docx_text(path), False
            elif mime_type == "text/csv":
                return self._extract_csv_text(path), False
            elif mime_type.startswith("text/"):
                return path.read_text(encoding="utf-8", errors="replace"), False
            else:
                # Try to read as text, fall back to description
                try:
                    return path.read_text(encoding="utf-8", errors="replace"), False
                except Exception:
                    return f"[Binary file: {path.name}, {path.stat().st_size} bytes]", False
        except Exception as e:
            return f"[Could not extract text: {e}]", False

    def _extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF using pypdf or pdfplumber."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text = []
            for page in reader.pages[:50]:  # Limit pages
                text.append(page.extract_text() or "")
            return "\n\n".join(text)
        except ImportError:
            pass

        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                text = []
                for page in pdf.pages[:50]:  # Limit pages
                    text.append(page.extract_text() or "")
                return "\n\n".join(text)
        except ImportError:
            pass

        return f"[PDF file: {path.name} - install pypdf or pdfplumber to extract text]"

    def _extract_docx_text(self, path: Path) -> str:
        """Extract text from Word documents."""
        try:
            from docx import Document
            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n\n".join(paragraphs)
        except ImportError:
            return f"[Word document: {path.name} - install python-docx to extract text]"

    def _extract_csv_text(self, path: Path) -> str:
        """Extract preview from CSV."""
        try:
            import pandas as pd
            df = pd.read_csv(str(path), nrows=100)  # First 100 rows
            return f"CSV with {len(df.columns)} columns: {', '.join(df.columns)}\n\nFirst rows:\n{df.head(20).to_string()}"
        except ImportError:
            # Fall back to raw text
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.split("\n")[:50]
            return "\n".join(lines)

    def _build_context(
        self,
        user_context: dict | None,
        existing_projects: list[str] | None,
    ) -> dict:
        """Build context for the extraction prompt."""
        context = {
            "user_preferences": "",
            "current_objectives": "",
            "recent_activity": "",
            "existing_projects": existing_projects or [],
        }

        if user_context:
            context["user_preferences"] = user_context.get("preferences", "")
            context["current_objectives"] = user_context.get("objectives", "")
            context["recent_activity"] = user_context.get("recent_activity", "")

        return context

    def _extract_with_llm(
        self,
        content: str | bytes,
        is_image: bool,
        mime_type: str,
        filename: str,
        context: dict,
    ) -> dict:
        """Run LLM extraction on content."""
        system_prompt = """You are extracting structured knowledge from a document for an AI agent's memory system.

Your goal is to extract ALL discrete facts, entities, and relationships that could be useful for later retrieval.

IMPORTANT EXTRACTION GUIDELINES:
1. Extract EVERY piece of factual information as a separate fact triplet
2. Be comprehensive - extract dates, numbers, names, relationships, events, metrics
3. Use natural, complete sentences for fact_text
4. Determine which project folder this document best belongs to based on content and existing projects
5. For images: describe what you see and extract any visible text, data, or information

FACT TYPES:
- relation: Relationship between entities (John works at Acme)
- attribute: Property of an entity (John's age is 35)
- event: Something that happened (Company was founded in 2020)
- state: Current status (Project status is active)
- metric: Numerical data (Q4 revenue was $4.2M)

Return valid JSON matching this schema:
{
    "summary": "2-3 sentence summary of the document",
    "facts": [
        {
            "subject": "entity name (required)",
            "predicate": "verb or relationship",
            "object": "entity name OR literal value",
            "object_type": "entity|value|text",
            "fact_type": "relation|attribute|event|state|metric",
            "fact_text": "Full natural sentence expressing this fact",
            "confidence": 0.0-1.0,
            "valid_from": "ISO date if known, null otherwise",
            "valid_to": "ISO date if known, null if current",
            "mentioned_entities": ["other entities mentioned in this fact's context"]
        }
    ],
    "entities": [
        {
            "name": "canonical name",
            "type": "person|org|tool|concept|location|file",
            "description": "brief description",
            "aliases": ["alternative names"]
        }
    ],
    "suggested_project": "project folder name (use existing if appropriate, or suggest new)",
    "tags": ["relevant", "tags", "for", "categorization"]
}"""

        # Build user message with file content
        user_message_parts = []

        # Add context
        user_message_parts.append(f"""CONTEXT:
User preferences: {context['user_preferences'] or 'None specified'}
Current objectives: {context['current_objectives'] or 'None specified'}
Recent activity: {context['recent_activity'] or 'None specified'}
Existing project folders: {', '.join(context['existing_projects']) if context['existing_projects'] else 'None yet'}

FILE INFORMATION:
Filename: {filename}
Type: {mime_type}
""")

        # Build message content
        if is_image and isinstance(content, bytes):
            # Use vision for images
            base64_image = base64.b64encode(content).decode("utf-8")
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(user_message_parts) + "\n\nAnalyze this image and extract all facts, entities, and information:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_image,
                        }
                    }
                ]
            }]
        else:
            # Text content
            text_content = content if isinstance(content, str) else content.decode("utf-8", errors="replace")
            # Truncate if too long
            if len(text_content) > 100000:
                text_content = text_content[:100000] + "\n\n[... truncated ...]"

            user_message_parts.append(f"DOCUMENT CONTENT:\n{text_content}")
            user_message_parts.append("\nExtract all facts, entities, and information from this document.")

            messages = [{
                "role": "user",
                "content": "\n".join(user_message_parts)
            }]

        # Call LLM
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=messages,
        )

        # Parse response
        response_text = response.content[0].text

        # Calculate costs
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)

        # Extract JSON from response
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)
            else:
                result = {
                    "summary": response_text[:500],
                    "facts": [],
                    "entities": [],
                    "suggested_project": "Inbox",
                    "tags": [],
                }
        except json.JSONDecodeError:
            result = {
                "summary": response_text[:500],
                "facts": [],
                "entities": [],
                "suggested_project": "Inbox",
                "tags": [],
            }

        # Add extraction metrics
        result["extraction_call"] = {
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "duration_ms": 0,  # Will be set by caller
        }

        return result

    def decide_extraction_timing(
        self,
        file_info: dict,
        user_context: dict | None = None,
    ) -> str:
        """
        Let LLM decide whether to extract immediately, in background, or on-demand.

        Returns: "immediate" | "background" | "on_demand"
        """
        # Quick heuristics first
        size_bytes = file_info.get("size_bytes", 0)
        mime_type = file_info.get("mime_type", "")

        # Small files: immediate
        if size_bytes < 50000:  # 50KB
            return "immediate"

        # Very large files: on_demand
        if size_bytes > 10000000:  # 10MB
            return "on_demand"

        # Check context for urgency
        if user_context:
            objectives = user_context.get("objectives", "")
            recent = user_context.get("recent_activity", "")

            # If user seems to be asking about this file, extract now
            if any(keyword in objectives.lower() or keyword in recent.lower()
                   for keyword in ["analyze", "read", "what does", "summary", "extract"]):
                return "immediate"

        # Default: background processing
        return "background"
