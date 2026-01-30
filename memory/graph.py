"""
Graph - Entities, edges, and their relationships.

The graph is derived from events via LLM extraction. It provides:
- Entities: People, organizations, topics, concepts
- Edges: Relationships between entities
- Efficient traversal and querying

Design: Simple in-memory graph with JSON persistence.
Entity and edge types are free-form and get clustered automatically.
"""

import json
import threading
from pathlib import Path
from typing import Iterator
from .models import Entity, Edge


class Graph:
    """
    Knowledge graph storage with entity and edge management.

    Design principles:
    - Free-form types that get clustered
    - Deduplication by name similarity
    - Efficient neighbor traversal
    """

    def __init__(self, storage_path: Path | None = None):
        self._entities: dict[str, Entity] = {}
        self._edges: dict[str, Edge] = {}

        # Indices for fast lookup
        self._by_type: dict[str, set[str]] = {}  # type -> entity_ids
        self._by_name: dict[str, str] = {}  # lowercase name -> entity_id
        self._by_alias: dict[str, str] = {}  # lowercase alias -> entity_id
        self._outgoing: dict[str, set[str]] = {}  # entity_id -> edge_ids
        self._incoming: dict[str, set[str]] = {}  # entity_id -> edge_ids

        # Type clustering (free-form -> canonical)
        self._type_clusters: dict[str, str] = {}  # e.g., "venture capitalist" -> "person"

        self._lock = threading.Lock()
        self._storage_path = storage_path

        if storage_path:
            self._load()

    def _load(self):
        """Load graph from disk."""
        if not self._storage_path or not self._storage_path.exists():
            return

        with open(self._storage_path, "r") as f:
            data = json.load(f)

        for ed in data.get("entities", []):
            entity = Entity.from_dict(ed)
            self._index_entity(entity)

        for ed in data.get("edges", []):
            edge = Edge.from_dict(ed)
            self._index_edge(edge)

        self._type_clusters = data.get("type_clusters", {})

    def _save(self):
        """Save graph to disk."""
        if not self._storage_path:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "type_clusters": self._type_clusters,
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _index_entity(self, entity: Entity):
        """Add entity to indices."""
        self._entities[entity.id] = entity

        # Index by type
        etype = entity.type.lower()
        if etype not in self._by_type:
            self._by_type[etype] = set()
        self._by_type[etype].add(entity.id)

        # Index by name and aliases
        self._by_name[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            self._by_alias[alias.lower()] = entity.id

        # Initialize edge indices
        if entity.id not in self._outgoing:
            self._outgoing[entity.id] = set()
        if entity.id not in self._incoming:
            self._incoming[entity.id] = set()

    def _index_edge(self, edge: Edge):
        """Add edge to indices."""
        self._edges[edge.id] = edge
        self._outgoing.setdefault(edge.source_id, set()).add(edge.id)
        self._incoming.setdefault(edge.target_id, set()).add(edge.id)

    # === Entity Operations ===

    def add_entity(self, entity: Entity) -> Entity:
        """
        Add or merge an entity.

        If an entity with the same name exists, merges attributes and aliases.
        """
        with self._lock:
            existing_id = self._by_name.get(entity.name.lower())
            if existing_id:
                existing = self._entities[existing_id]
                # Merge
                existing.aliases = list(set(existing.aliases + entity.aliases))
                existing.attributes.update(entity.attributes)
                existing.source_events = list(
                    set(existing.source_events + entity.source_events)
                )
                existing.updated_at = entity.updated_at
                self._save()
                return existing

            self._index_entity(entity)
            self._save()
            return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def find_entity(self, name: str) -> Entity | None:
        """Find entity by name or alias."""
        name_lower = name.lower()
        entity_id = self._by_name.get(name_lower) or self._by_alias.get(name_lower)
        return self._entities.get(entity_id) if entity_id else None

    def search_entities(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """
        Search entities by name/type.

        Args:
            query: Text to match in name or aliases
            types: Filter by entity types
            limit: Maximum results
        """
        results = []
        query_lower = query.lower() if query else None

        for entity in self._entities.values():
            # Type filter
            if types:
                canonical_type = self._type_clusters.get(entity.type.lower(), entity.type.lower())
                if canonical_type not in [t.lower() for t in types]:
                    continue

            # Name filter
            if query_lower:
                name_match = query_lower in entity.name.lower()
                alias_match = any(query_lower in a.lower() for a in entity.aliases)
                if not (name_match or alias_match):
                    continue

            results.append(entity)
            if len(results) >= limit:
                break

        return results

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a type (including clustered types)."""
        entity_ids = self._by_type.get(entity_type.lower(), set())
        # Also include types that cluster to this type
        for raw_type, canonical in self._type_clusters.items():
            if canonical.lower() == entity_type.lower():
                entity_ids |= self._by_type.get(raw_type, set())
        return [self._entities[eid] for eid in entity_ids]

    # === Edge Operations ===

    def add_edge(self, edge: Edge) -> Edge:
        """
        Add or update an edge.

        If an edge with same source/target/type exists, merges attributes.
        """
        with self._lock:
            # Check for existing
            for existing in self._edges.values():
                if (
                    existing.source_id == edge.source_id
                    and existing.target_id == edge.target_id
                    and existing.type.lower() == edge.type.lower()
                ):
                    existing.attributes.update(edge.attributes)
                    existing.source_events = list(
                        set(existing.source_events + edge.source_events)
                    )
                    existing.updated_at = edge.updated_at
                    self._save()
                    return existing

            self._index_edge(edge)
            self._save()
            return edge

    def get_edge(self, edge_id: str) -> Edge | None:
        """Get edge by ID."""
        return self._edges.get(edge_id)

    def get_edges(
        self,
        entity_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> list[Edge]:
        """
        Get edges connected to an entity.

        Args:
            entity_id: The entity to find edges for
            direction: "outgoing", "incoming", or "both"
            edge_types: Filter by edge types
        """
        edge_ids = set()
        if direction in ("outgoing", "both"):
            edge_ids |= self._outgoing.get(entity_id, set())
        if direction in ("incoming", "both"):
            edge_ids |= self._incoming.get(entity_id, set())

        edges = [self._edges[eid] for eid in edge_ids]

        if edge_types:
            types_lower = [t.lower() for t in edge_types]
            edges = [e for e in edges if e.type.lower() in types_lower]

        return edges

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> list[tuple[Entity, Edge]]:
        """
        Get neighboring entities with their connecting edges.

        Returns list of (entity, edge) tuples.
        """
        results = []
        edges = self.get_edges(entity_id, direction, edge_types)

        for edge in edges:
            neighbor_id = edge.target_id if edge.source_id == entity_id else edge.source_id
            neighbor = self._entities.get(neighbor_id)
            if neighbor:
                results.append((neighbor, edge))

        return results

    # === Type Clustering ===

    def set_type_cluster(self, raw_type: str, canonical_type: str):
        """Map a free-form type to a canonical type."""
        with self._lock:
            self._type_clusters[raw_type.lower()] = canonical_type.lower()
            self._save()

    def get_canonical_type(self, raw_type: str) -> str:
        """Get the canonical type for a raw type."""
        return self._type_clusters.get(raw_type.lower(), raw_type.lower())

    # === Utilities ===

    def all_entities(self) -> Iterator[Entity]:
        """Iterate over all entities."""
        for entity in self._entities.values():
            yield entity

    def all_edges(self) -> Iterator[Edge]:
        """Iterate over all edges."""
        for edge in self._edges.values():
            yield edge

    def entity_count(self) -> int:
        return len(self._entities)

    def edge_count(self) -> int:
        return len(self._edges)

    def get_entity_types(self) -> list[str]:
        """Get all unique entity types."""
        return list(self._by_type.keys())

    def get_edge_types(self) -> list[str]:
        """Get all unique edge types."""
        types = set()
        for edge in self._edges.values():
            types.add(edge.type.lower())
        return list(types)
