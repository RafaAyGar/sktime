# -*- coding: utf-8 -*-
"""Vanilla Classifier.

A vanilla classifier is any standard classification method fed with transposed time series data, i.e. the time series values are treated as features.
"""

__author__ = ["Rafael Ayllon"]
__all__ = ["VanillaClassifier"]

import numpy as np
from sklearn.linear_model import LogisticRegression
from sktime.classification.base import BaseClassifier

class VanillaClassifier(BaseClassifier):
    """Vanilla Classifier.

    This classifier receives a non time series classifier (e.g. LogisticRegression or DecisionTreeClassifier), and applies it to the transposed time series data, i.e. the time series values are treated as features. The purpose of this classifier is to compare the performance of the vanilla method with the performance of any time series classifier.

    Parameters
    ----------
    base_estimator : object, default=None
        The base non time series estimator to fit on transposed data.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:multithreading": True,
        "capability:missing_values": True,
    }

    _estimator_type = "classifier"

    def __init__(
        self,
        base_estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.base_estimator = base_estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        super(VanillaClassifier, self).__init__()
    
    def _fit(self, X, y) -> BaseClassifier:
        """Fit classifier to training data.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, 1, series_length]
            The training input samples.
        y : 1D np.array of shape = [n_instances]
            The target values (class labels) as integers or strings.

        Returns
        -------
        self : object
            Returns self.
        """
        X_t = self.transform_data(X)

        if self.base_estimator is None:
            self.base_estimator = LogisticRegression()
        self.base_estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class labels for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, 1, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.

        """
        X_t = self.transform_data(X)
        return self.base_estimator.predict(X_t)

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        X_t = self.transform_data(X)
        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X_t)
        else:
            preds = self.base_estimator.predict(X_t)
            dists = np.zeros((X.shape[0], self.n_classes_))
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
        return dists

    def transform_data(self, X) -> np.ndarray:
        """Transform the data.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, 1, series_length]
            The data to transform.

        Returns
        -------
        X_t : array-like, shape = [n_instances, series_length]
            Transformed data.
        """
        X = X.reshape(X.shape[0], X.shape[2])
        return X