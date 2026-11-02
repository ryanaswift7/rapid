from typing import Any, List, Literal, Optional, Union

from pydantic import computed_field, field_validator, model_validator

from rapid.schemas.configBase import ConfigBase


class LMEvalTasks(ConfigBase):
    names: List[str]
    sample_limit: Optional[int] = None
    force_rerun: bool = False


TASK_PROPERTIES = {
    "boolq": {"task_type": "binary", "test_available": False, "n_train_samples": 9_427},
    "winogrande": {
        "task_type": "binary",
        "test_available": False,
        "n_train_samples": 40_398,
    },
    "piqa": {"task_type": "binary", "test_available": False, "n_train_samples": 16_113},
    "arc_challenge": {
        "task_type": "multiclass",
        "test_available": True,
        "n_train_samples": 1_119,
    },
    "arc_easy": {
        "task_type": "multiclass",
        "test_available": True,
        "n_train_samples": 2_251,
    },
    "hellaswag": {
        "task_type": "multiclass",
        "test_available": False,
        "n_train_samples": 39_905,
    },
    "sciq": {
        "task_type": "multiclass",
        "test_available": True,
        "n_train_samples": 11_679,
    },
}


class TaskConfig(ConfigBase):
    dataset_name: str
    subset: Optional[str] = None


class TaskInstance(ConfigBase):
    name: str
    dataset_name: str
    subset: Optional[str] = None
    num_fewshot: int = 0
    batch_size: Union[int, str] = "auto"

    @field_validator("name")
    @classmethod
    def validate_name_exists(cls, v: str) -> str:
        if v not in TASK_PROPERTIES:
            raise ValueError(
                f"Task '{v}' is not supported. Please add its properties "
                f"to the TASK_PROPERTIES dictionary."
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def purge_computed_fields(cls, data: Any) -> Any:
        # Dynamically remove ANY field that is defined as a @computed_field
        # NOTE: necessary to resolve conflicts from load_checkpoint
        if isinstance(data, dict):
            for field_name in cls.model_computed_fields.keys():
                data.pop(field_name, None)
        return data

    @computed_field
    @property
    def task_type(self) -> Literal["binary", "multiclass"]:
        """Authoritatively sets the task_type based on the task name."""
        if self.name in TASK_PROPERTIES:
            return TASK_PROPERTIES[self.name]["task_type"]
        else:
            # Raising an error in a computed field prevents model instantiation.
            raise ValueError(
                f"Task '{self.name}' is not supported. Please add its properties "
                f"to the TASK_PROPERTIES dictionary."
            )

    @computed_field
    @property
    def test_available(self) -> Optional[bool]:
        """Authoritatively sets test_available based on the task name."""
        return TASK_PROPERTIES.get(self.name, {}).get("test_available")
        # Note: The ValueError from task_type ensures self.name is valid before this runs.

    @computed_field
    @property
    def ninety_percent(self) -> int:
        """
        Returns the 90% sample count.

        Raises:
            ValueError: If test_available is True or the task name is missing.
            KeyError: If n_train_samples is not defined for this task.
            TypeError: If n_train_samples is not a numeric type.
        """
        if self.test_available:
            raise ValueError(
                f"Cannot calculate ninety_percent for {self.name}: test is available."
            )

        # 1. Ensure the task exists in your registry
        if self.name not in TASK_PROPERTIES:
            raise KeyError(f"Task '{self.name}' not found in TASK_PROPERTIES.")

        task_meta = TASK_PROPERTIES[self.name]

        # 2. Ensure the specific key exists
        if "n_train_samples" not in task_meta:
            raise KeyError(f"Task '{self.name}' is missing the 'n_train_samples' key.")

        n_train = task_meta["n_train_samples"]

        # 3. Type Guard: Ensure math is performed on a valid number
        if not isinstance(n_train, (int, float)):
            raise TypeError(
                f"Expected int/float for 'n_train_samples' in task '{self.name}', "
                f"got {type(n_train).__name__}."
            )

        return int(round(0.9 * n_train))

    @computed_field
    @property
    def n_train_samples(self) -> int:
        """
        Authoritatively retrieves the n_train_samples count.

        Raises:
            KeyError: If the task name or n_train_samples key is missing.
            TypeError: If the value found is not an integer.
        """
        # 1. Check if task exists
        if self.name not in TASK_PROPERTIES:
            raise KeyError(
                f"Property lookup failed: Task '{self.name}' is not defined in TASK_PROPERTIES."
            )

        task_data = TASK_PROPERTIES[self.name]

        # 2. Check if the specific key exists
        if "n_train_samples" not in task_data:
            raise KeyError(
                f"Configuration error: 'n_train_samples' missing for task '{self.name}'."
            )

        val = task_data["n_train_samples"]

        # 3. Type Guard: Ensure it is an int (and not None or a string)
        if not isinstance(val, int):
            raise TypeError(
                f"Type error for task '{self.name}': 'n_train_samples' must be an int, "
                f"but found {type(val).__name__}."
            )

        return val
