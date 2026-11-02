from typing import Any, ClassVar, Dict, Set

from pydantic import BaseModel, model_validator


class ConfigBase(BaseModel):
    """
    Abstract base class for all configuration objects.
    Provides standardized methods for serialization, comparison, and hashing
    using Pydantic's built-in functionality.
    """

    # Pydantic V2+ uses a `model_config` dictionary instead of a class `Config`
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    # Subclasses should override this set to specify fields to exclude.
    # This replaces the need to manually implement `to_dict()` everywhere.
    # Use ClassVar to tell Pylance/Pydantic this isn't an instance field
    _heavy_fields_to_exclude: ClassVar[Set[str]] = set()

    @model_validator(mode="before")
    @classmethod
    def strip_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # cls.model_computed_fields is a dict of all @computed_field definitions
            computed_names = cls.model_computed_fields.keys()
            for field in computed_names:
                data.pop(field, None)
        return data

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Return a standardized dictionary representation for configuration purposes.
        This automatically excludes fields defined in `_heavy_fields_to_exclude`.
        """
        return self.model_dump(exclude=self._heavy_fields_to_exclude)

    @property
    def config_json(self) -> str:
        """Property that returns a sorted JSON representation for hashing."""
        return self.model_dump_json(
            exclude=self._heavy_fields_to_exclude,
            indent=4,
        )

    def __hash__(self) -> int:
        """Make objects hashable based on their configuration."""
        return hash(self.config_json)

    def __eq__(self, other: object) -> bool:
        """Compare objects based on their configuration."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.config_json == other.config_json
