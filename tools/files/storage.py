"""
File Storage Manager

Handles file storage with project-based organization.
Files are stored at ~/.babyagi/files/{project}/{filename}
"""

import os
import shutil
from pathlib import Path
from typing import BinaryIO
from datetime import datetime
import mimetypes
import hashlib


class FileStorage:
    """Manages file storage with project-based organization."""

    def __init__(self, base_path: str = "~/.babyagi/files"):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_project_path(self, project: str) -> Path:
        """Get or create a project directory."""
        # Sanitize project name
        safe_name = self._sanitize_name(project)
        project_path = self.base_path / safe_name
        project_path.mkdir(parents=True, exist_ok=True)
        return project_path

    def list_projects(self) -> list[str]:
        """List all project folders."""
        if not self.base_path.exists():
            return []
        return [
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def save_file(
        self,
        content: bytes | str,
        filename: str,
        project: str,
        overwrite: bool = False,
    ) -> dict:
        """
        Save content to a file in the specified project.

        Returns:
            {
                "path": absolute path to file,
                "project": project name,
                "filename": final filename,
                "size_bytes": file size,
                "mime_type": detected mime type,
                "hash": SHA256 hash of content,
            }
        """
        project_path = self.get_project_path(project)
        safe_filename = self._sanitize_name(filename)

        # Handle filename conflicts
        file_path = project_path / safe_filename
        if file_path.exists() and not overwrite:
            # Add timestamp to filename
            stem = file_path.stem
            suffix = file_path.suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{stem}_{timestamp}{suffix}"
            file_path = project_path / safe_filename

        # Write content
        if isinstance(content, str):
            content = content.encode("utf-8")

        file_path.write_bytes(content)

        # Calculate hash
        file_hash = hashlib.sha256(content).hexdigest()

        # Detect mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return {
            "path": str(file_path.absolute()),
            "project": project,
            "filename": safe_filename,
            "size_bytes": len(content),
            "mime_type": mime_type or "application/octet-stream",
            "hash": file_hash,
        }

    def save_file_from_path(
        self,
        source_path: str,
        project: str,
        new_filename: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Copy a file from source path to project storage."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        filename = new_filename or source.name
        content = source.read_bytes()

        return self.save_file(content, filename, project, overwrite)

    def read_file(self, path: str) -> bytes:
        """Read file content by path."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_bytes()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read file as text."""
        return self.read_file(path).decode(encoding)

    def delete_file(self, path: str) -> bool:
        """Delete a file."""
        file_path = Path(path)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def move_file(self, source_path: str, dest_project: str, new_filename: str | None = None) -> dict:
        """Move a file to a different project."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest_project_path = self.get_project_path(dest_project)
        filename = new_filename or source.name
        safe_filename = self._sanitize_name(filename)
        dest_path = dest_project_path / safe_filename

        shutil.move(str(source), str(dest_path))

        # Get file info
        content = dest_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        mime_type, _ = mimetypes.guess_type(str(dest_path))

        return {
            "path": str(dest_path.absolute()),
            "project": dest_project,
            "filename": safe_filename,
            "size_bytes": dest_path.stat().st_size,
            "mime_type": mime_type or "application/octet-stream",
            "hash": file_hash,
        }

    def list_files(self, project: str | None = None) -> list[dict]:
        """
        List files in a project or all projects.

        Returns list of file info dicts.
        """
        files = []

        if project:
            projects = [project]
        else:
            projects = self.list_projects()

        for proj in projects:
            project_path = self.base_path / self._sanitize_name(proj)
            if not project_path.exists():
                continue

            for file_path in project_path.iterdir():
                if file_path.is_file() and not file_path.name.startswith("."):
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    stat = file_path.stat()
                    files.append({
                        "path": str(file_path.absolute()),
                        "project": proj,
                        "filename": file_path.name,
                        "size_bytes": stat.st_size,
                        "mime_type": mime_type or "application/octet-stream",
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })

        return files

    def get_file_info(self, path: str) -> dict | None:
        """Get info about a specific file."""
        file_path = Path(path)
        if not file_path.exists():
            return None

        # Try to determine project from path
        try:
            rel_path = file_path.relative_to(self.base_path)
            project = rel_path.parts[0] if rel_path.parts else "unknown"
        except ValueError:
            project = "external"

        mime_type, _ = mimetypes.guess_type(str(file_path))
        stat = file_path.stat()

        content = file_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        return {
            "path": str(file_path.absolute()),
            "project": project,
            "filename": file_path.name,
            "size_bytes": stat.st_size,
            "mime_type": mime_type or "application/octet-stream",
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "hash": file_hash,
        }

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a filename or project name."""
        # Replace problematic characters
        safe = name.replace("/", "_").replace("\\", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "._- ")
        safe = safe.strip()
        return safe or "unnamed"
