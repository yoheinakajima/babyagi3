"""
Files Tool - Unified file management for the agent.

Provides a single elegant interface for all file operations:
- Save files (with automatic extraction and organization)
- Read files
- Create files (CSV, images, Word docs, PDFs)
- List and search files
- Move and organize files

Files are stored at ~/.babyagi/files/{project}/{filename}
Each file becomes an entity in the knowledge graph with facts extracted.
"""

import base64
import json
from pathlib import Path
from typing import Any
from datetime import datetime

from tools import tool, tool_error

from .storage import FileStorage
from .processor import DocumentProcessor
from .creators import (
    create_csv,
    create_image,
    combine_images,
    create_word_doc,
    create_pdf,
)


# Global instances (initialized on first use)
_storage: FileStorage | None = None
_processor: DocumentProcessor | None = None


def _get_storage() -> FileStorage:
    global _storage
    if _storage is None:
        _storage = FileStorage()
    return _storage


def _get_processor() -> DocumentProcessor:
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


@tool(packages=["anthropic"])
def files(
    action: str,
    path: str | None = None,
    project: str | None = None,
    content: str | None = None,
    filename: str | None = None,
    query: str | None = None,
    file_type: str | None = None,
    data: list | dict | None = None,
    options: dict | None = None,
) -> dict:
    """
    Unified file management - store, organize, search, and create files.

    Actions:
        save     - Save content or file to storage (extracts facts, auto-organizes)
        read     - Read file content
        list     - List files (optionally in a project)
        search   - Semantic search across file summaries
        move     - Move file to different project
        delete   - Delete a file
        create   - Create a new file (CSV, image, Word doc, PDF)
        projects - List all project folders
        info     - Get info about a specific file

    Args:
        action: The action to perform
        path: File path (for read, move, delete, info)
        project: Project folder name (for save, list, move)
        content: Content to save (string or base64 for binary)
        filename: Filename for saving or creating
        query: Search query (for search action)
        file_type: Type for create action (csv, image, docx, pdf)
        data: Data for create action (varies by type)
        options: Additional options depending on action

    Returns:
        Action-specific result dict

    Examples:
        # Save a file
        files(action="save", content="report text", filename="report.txt", project="Q4 Reports")

        # Create a CSV
        files(action="create", file_type="csv", data=[{"name": "John", "age": 30}], filename="people.csv", project="Data")

        # List files in a project
        files(action="list", project="Q4 Reports")

        # Search for files
        files(action="search", query="revenue analysis")
    """
    storage = _get_storage()
    processor = _get_processor()
    options = options or {}

    try:
        if action == "save":
            return _action_save(storage, processor, content, filename, project, options)

        elif action == "read":
            return _action_read(storage, path)

        elif action == "list":
            return _action_list(storage, project)

        elif action == "search":
            return _action_search(storage, query, options)

        elif action == "move":
            return _action_move(storage, path, project, filename)

        elif action == "delete":
            return _action_delete(storage, path)

        elif action == "create":
            return _action_create(storage, file_type, data, filename, project, options)

        elif action == "projects":
            return _action_projects(storage)

        elif action == "info":
            return _action_info(storage, path)

        else:
            return tool_error(
                f"Unknown action: {action}",
                fix="Use one of: save, read, list, search, move, delete, create, projects, info"
            )

    except FileNotFoundError as e:
        return tool_error(str(e), fix="Check that the file path is correct")
    except Exception as e:
        return tool_error(f"File operation failed: {e}")


def _action_save(
    storage: FileStorage,
    processor: DocumentProcessor,
    content: str | None,
    filename: str | None,
    project: str | None,
    options: dict,
) -> dict:
    """Save content to storage with extraction."""
    if not content:
        return tool_error("Content is required for save action")
    if not filename:
        return tool_error("Filename is required for save action")

    # Decode base64 if it looks like it
    if content.startswith("data:"):
        # Data URL format: data:image/png;base64,xxxxx
        parts = content.split(",", 1)
        if len(parts) == 2:
            content = base64.b64decode(parts[1])
    elif _is_base64(content):
        try:
            content = base64.b64decode(content)
        except Exception:
            pass  # Not base64, treat as text

    # Get existing projects for context
    existing_projects = storage.list_projects()

    # Build user context from options
    user_context = {
        "preferences": options.get("preferences", ""),
        "objectives": options.get("objectives", ""),
        "recent_activity": options.get("recent_activity", ""),
    }

    # Process the content to extract facts and determine project
    if isinstance(content, bytes):
        mime_type = options.get("mime_type", _guess_mime_type(filename))
        extraction = processor.process_content(
            content, filename, mime_type, user_context, existing_projects
        )
    else:
        extraction = processor.process_content(
            content, filename, "text/plain", user_context, existing_projects
        )

    # Use suggested project if not specified
    if not project:
        project = extraction.get("suggested_project", "Inbox")

    # Save the file
    if isinstance(content, bytes):
        file_info = storage.save_file(content, filename, project)
    else:
        file_info = storage.save_file(content.encode("utf-8"), filename, project)

    return {
        "success": True,
        "path": file_info["path"],
        "project": project,
        "filename": file_info["filename"],
        "size_bytes": file_info["size_bytes"],
        "summary": extraction.get("summary", ""),
        "facts_extracted": len(extraction.get("facts", [])),
        "entities_found": len(extraction.get("entities", [])),
        "tags": extraction.get("tags", []),
        "extraction": extraction,  # Full extraction for memory integration
    }


def _action_read(storage: FileStorage, path: str | None) -> dict:
    """Read file content."""
    if not path:
        return tool_error("Path is required for read action")

    content = storage.read_file(path)
    file_info = storage.get_file_info(path)

    # Try to decode as text
    try:
        text_content = content.decode("utf-8")
        is_binary = False
    except UnicodeDecodeError:
        text_content = None
        is_binary = True

    result = {
        "success": True,
        "path": path,
        "filename": file_info["filename"] if file_info else Path(path).name,
        "size_bytes": len(content),
        "is_binary": is_binary,
    }

    if is_binary:
        result["content_base64"] = base64.b64encode(content).decode("utf-8")
    else:
        result["content"] = text_content

    return result


def _action_list(storage: FileStorage, project: str | None) -> dict:
    """List files in a project or all projects."""
    files = storage.list_files(project)

    return {
        "success": True,
        "project": project,
        "count": len(files),
        "files": files,
    }


def _action_search(storage: FileStorage, query: str | None, options: dict) -> dict:
    """Search for files (placeholder - needs memory integration)."""
    if not query:
        return tool_error("Query is required for search action")

    # For now, simple filename/project matching
    # TODO: Integrate with fact embeddings for semantic search
    all_files = storage.list_files()
    query_lower = query.lower()

    matches = []
    for f in all_files:
        if (query_lower in f["filename"].lower() or
            query_lower in f["project"].lower()):
            matches.append(f)

    return {
        "success": True,
        "query": query,
        "count": len(matches),
        "files": matches,
        "note": "Semantic search will be available when integrated with memory system",
    }


def _action_move(
    storage: FileStorage,
    path: str | None,
    project: str | None,
    new_filename: str | None,
) -> dict:
    """Move file to a different project."""
    if not path:
        return tool_error("Path is required for move action")
    if not project:
        return tool_error("Project is required for move action")

    file_info = storage.move_file(path, project, new_filename)

    return {
        "success": True,
        "new_path": file_info["path"],
        "project": project,
        "filename": file_info["filename"],
    }


def _action_delete(storage: FileStorage, path: str | None) -> dict:
    """Delete a file."""
    if not path:
        return tool_error("Path is required for delete action")

    deleted = storage.delete_file(path)

    return {
        "success": deleted,
        "path": path,
        "deleted": deleted,
    }


def _action_create(
    storage: FileStorage,
    file_type: str | None,
    data: list | dict | None,
    filename: str | None,
    project: str | None,
    options: dict,
) -> dict:
    """Create a new file of specified type."""
    if not file_type:
        return tool_error("file_type is required for create action")

    file_type = file_type.lower()

    try:
        if file_type == "csv":
            if not data:
                return tool_error("data is required for CSV creation")
            headers = options.get("headers")
            content, suggested_filename = create_csv(data, headers, filename)

        elif file_type == "image":
            width = options.get("width", 800)
            height = options.get("height", 600)
            background = options.get("background", "white")
            elements = data if isinstance(data, list) else []
            format = options.get("format", "PNG")
            content, suggested_filename = create_image(
                width, height, background, elements, filename, format
            )

        elif file_type == "combine_images":
            if not data or not isinstance(data, list):
                return tool_error("data must be a list of image paths for combine_images")
            layout = options.get("layout", "horizontal")
            spacing = options.get("spacing", 10)
            background = options.get("background", "white")
            content, suggested_filename = combine_images(
                data, layout, spacing, background, filename
            )

        elif file_type in ("docx", "word"):
            if not data and not isinstance(data, (str, list)):
                return tool_error("data (string or list of elements) is required for Word doc creation")
            title = options.get("title")
            content_data = data if data else ""
            content, suggested_filename = create_word_doc(content_data, title, filename)

        elif file_type == "pdf":
            if not data and not isinstance(data, (str, list)):
                return tool_error("data (string or list of elements) is required for PDF creation")
            title = options.get("title")
            content_data = data if data else ""
            content, suggested_filename = create_pdf(content_data, title, filename)

        else:
            return tool_error(
                f"Unknown file_type: {file_type}",
                fix="Use one of: csv, image, combine_images, docx, word, pdf"
            )

        # Use suggested filename if none provided
        final_filename = filename or suggested_filename

        # Use default project if none specified
        final_project = project or "Created Files"

        # Save the created file
        file_info = storage.save_file(content, final_filename, final_project)

        return {
            "success": True,
            "path": file_info["path"],
            "project": final_project,
            "filename": file_info["filename"],
            "size_bytes": file_info["size_bytes"],
            "file_type": file_type,
        }

    except ImportError as e:
        return tool_error(
            f"Missing required package: {e}",
            fix="Install the required package to enable this file type"
        )


def _action_projects(storage: FileStorage) -> dict:
    """List all project folders."""
    projects = storage.list_projects()

    # Count files in each project
    project_info = []
    for proj in projects:
        files = storage.list_files(proj)
        project_info.append({
            "name": proj,
            "file_count": len(files),
        })

    return {
        "success": True,
        "count": len(projects),
        "projects": project_info,
    }


def _action_info(storage: FileStorage, path: str | None) -> dict:
    """Get info about a specific file."""
    if not path:
        return tool_error("Path is required for info action")

    info = storage.get_file_info(path)
    if not info:
        return tool_error(f"File not found: {path}")

    return {
        "success": True,
        **info,
    }


def _is_base64(s: str) -> bool:
    """Check if string looks like base64."""
    if len(s) < 100:
        return False
    try:
        # Check if it decodes and re-encodes to same
        decoded = base64.b64decode(s, validate=True)
        return len(decoded) > 0
    except Exception:
        return False


def _guess_mime_type(filename: str) -> str:
    """Guess mime type from filename."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"
