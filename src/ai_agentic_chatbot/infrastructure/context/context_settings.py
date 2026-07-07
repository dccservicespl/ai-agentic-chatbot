"""Context registry — loads and validates the contexts: block from config.yaml."""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel


class DbContextConfig(BaseModel):
    """Immutable configuration for a single database context."""

    context_id: str
    schema_name: str
    include_tables: List[str]
    system_prompt_path: str
    sql_examples_path: str
    router_prompt_path: str
    out_of_scope_message: str
    schema_dir: str
    vector_collection_name: str
    display_name: Optional[str] = None

    model_config = {"frozen": True, "extra": "forbid"}


class DbContextRegistry(BaseModel):
    """Registry of all configured database contexts, loaded from config.yaml."""

    contexts: Dict[str, DbContextConfig]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> "DbContextRegistry":
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            config_path = project_root / "config.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {}

        return cls._parse_config(config_data)

    @classmethod
    def _parse_config(cls, config_data: Dict) -> "DbContextRegistry":
        raw_contexts = config_data.get("contexts", {})
        contexts: Dict[str, DbContextConfig] = {}
        for context_id, ctx_data in raw_contexts.items():
            contexts[context_id] = DbContextConfig(
                context_id=context_id,
                **ctx_data,
            )
        return cls(contexts=contexts)

    def get_context(self, context_id: str) -> DbContextConfig:
        if context_id not in self.contexts:
            available = list(self.contexts.keys())
            raise KeyError(
                f"Context '{context_id}' not found. Available: {available}"
            )
        return self.contexts[context_id]

    def list_contexts(self) -> List[str]:
        return list(self.contexts.keys())


_registry: Optional[DbContextRegistry] = None


def get_context_registry() -> DbContextRegistry:
    """Return the global DbContextRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = DbContextRegistry.from_config_file()
    return _registry


def reload_context_registry() -> DbContextRegistry:
    """Force a re-read of config.yaml and reset the singleton."""
    global _registry
    _registry = DbContextRegistry.from_config_file()
    return _registry