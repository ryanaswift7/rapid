from typing import Optional

from rapid.schemas.configBase import ConfigBase


class DatasetConfig(ConfigBase):
    name: str
    subset: Optional[str] = None
    split: str = "train"
    shuffle: bool = False
    token_rename: bool = False
