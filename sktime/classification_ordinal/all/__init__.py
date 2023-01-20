# -*- coding: utf-8 -*-
"""All time series classifiers."""

__author__ = ["mloning"]
__all__ = [
    "ShapeletTransformClassifierOrdinal",
    "RocketClassifierOrdinal",
    "DrCIFOrdinal",
    "ArsenalOrdinal",
    "HIVECOTEV2Ordinal",
    "FreshPRINCEOrdinal"
]

from sktime.classification_ordinal.kernel_based import (
    RocketClassifierOrdinal,
    ArsenalOrdinal
)
from sktime.classification_ordinal.shapelet_based import ShapeletTransformClassifierOrdinal
from sktime.classification_ordinal.interval_based import DrCIFOrdinal
from sktime.classification_ordinal.feature_based import FreshPRINCEOrdinal
from sktime.classification_ordinal.hybrid import HIVECOTEV2Ordinal

