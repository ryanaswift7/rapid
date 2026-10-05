from typing import List, Optional

from kdfp.schemas.configBase import ConfigBase


class ExperimentFilters(ConfigBase):
    """
    Allows for string matching to select a subset of particular experiments to run
    from the full specified suite. Experiments are matched by their name. Useful for debugging.

    Note that combining include and exclude may result in unexpected behavior.

    Args:
        include:    Only experiment names that possess one of the strings specified here will be run.
        exclude:    All experiments *except* those that possess a matching string will be run.
    """

    include: Optional[List[str]]
    exclude: Optional[List[str]]
