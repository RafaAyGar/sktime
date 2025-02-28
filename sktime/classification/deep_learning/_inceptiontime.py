# -*- coding: utf-8 -*-
__author__ = "James Large"
__all__ = ["InceptionTimeClassifier"]

from tensorflow import keras

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks._inceptiontime import InceptionTimeNetwork
from sklearn.utils import check_random_state


class InceptionTimeClassifier(BaseDeepClassifier, InceptionTimeNetwork):
    """InceptionTime

    Parameters
    ----------
    nb_filters: int,
    use_residual: boolean,
    use_bottleneck: boolean,
    depth: int
    kernel_size: int, specifying the length of the 1D convolution
     window
    batch_size: int, the number of samples per gradient update.
    bottleneck_size: int,
    nb_epochs: int, the number of epochs to train the model
    callbacks: list of tf.keras.callbacks.Callback objects
    random_state: int, seed to any needed random actions
    verbose: boolean, whether to output extra information
    model_name: string, the name of this model for printing and
     file writing purposes
    model_save_directory: string, if not None; location to save
     the trained keras model in hdf5 format

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    """

    def __init__(
            self,
            nb_filters=32,
            use_residual=True,
            use_bottleneck=True,
            bottleneck_size=32,
            depth=6,
            kernel_size=41 - 1,
            batch_size=64,
            nb_epochs=1500,
            ensemble_size=5,
            callbacks=None,
            random_state=0,
            verbose=False,
            model_name="inception",
            model_save_directory=None,
    ):
        super(InceptionTimeClassifier, self).__init__(
            # model_name=model_name, model_save_directory=model_save_directory
        )

        self.verbose = verbose

        # predefined
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.ensemble_size = ensemble_size

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.model_name = model_name
        self.model_save_directory = model_save_directory
        self._is_fitted = False

    def build_model(self, input_shape, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output
             layer

        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if self.callbacks is None:
            self.callbacks = []

        if not any(
                isinstance(callback, keras.callbacks.ReduceLROnPlateau)
                for callback in self.callbacks
        ):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            self.callbacks.append(reduce_lr)

        return model

    def _fit(self, X, y):
        """
        Fit the classifier on the training set (X, y)

        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : array-like, shape = [n_instances]
            The validation class labels.

        Returns
        -------
        self : object
        """
        if self.callbacks is None:
            self._callbacks = []

        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        
        self.models = []
        for _ in range(self.ensemble_size):
            model = self.build_model(self.input_shape, self.n_classes_)
            if self.verbose:
                model.summary()
            self.history = model.fit(
                X,
                y_onehot,
                batch_size=self.batch_size,
                epochs=self.nb_epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
            )
            self.models.append(model)

        self._is_fitted = True
        
        return self
