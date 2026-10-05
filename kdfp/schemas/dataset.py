from typing import Optional

from kdfp.schemas.configBase import ConfigBase


class DatasetConfig(ConfigBase):
    name: str
    subset: Optional[str] = None
    split: str = "train"
    shuffle: bool = False
    token_rename: bool = False
