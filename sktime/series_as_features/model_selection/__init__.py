# -*- coding: utf-8 -*-
"""Model selection module."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
__all__ = [
    "SingleSplit",
    "PresplitFilesCV",
    "StratifiedCV",
]

from sktime.series_as_features.model_selection._split import (
    PresplitFilesCV,
    SingleSplit,
    StratifiedCV,
)
