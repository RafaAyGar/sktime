# -*- coding: utf-8 -*-
__author__ = "Rafael Ayllon"
__all__ = ["InceptionTimeOrdinalClassifier"]

import tensorflow as tf
from tensorflow import keras
from tensorflow.python import keras as keras_c

from sktime.classification.deep_learning.base import BaseDeepClassifier
from sktime.networks._inceptiontime import InceptionTimeNetwork
from sktime.utils import check_and_clean_data, \
    check_and_clean_validation_data

from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_sample_weight

from .activation_layers.clm import CLM


class InceptionTimeOrdinalClassifier(BaseDeepClassifier, InceptionTimeNetwork):
    """InceptionTimeOrdinal

    Ordinal implementation of the InceptionTime classifier [1], it substitutes the inception network softmax by a CLM activation layer [2].

    Parameters
    ----------
    nb_filters: int, default = 32
    use_residual: boolean, default = True
    use_bottleneck: boolean, default = True
    bottleneck_size: int, default = 32
    depth: int, default = 6
    kernel_size: int, specifying the length of the 1D convolution
     window, default = 41 - 1
    batch_size: int, the number of samples per gradient update, default = 64
    nb_epochs: int, the number of epochs to train the model, default = 1500
    ensemble_size: int, the number of models to train in the ensemble, default = 5
    callbacks: list of tf.keras.callbacks.Callback objects, default = None
    random_state: int, seed to any needed random actions, default = 0
    verbose: boolean, whether to output extra information, default = False
    model_name: string, the name of this model for printing and
     file writing purposes, default = "inception"
    model_save_directory: string, if not None; location to save
     the trained keras model in hdf5 format, default = None
    optimizer_lr: float, the learning rate for the optimizer, default = 0.001
    balanced_class_weights: boolean, whether to use balanced class weights in training, default = False

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020
    ..[2] Vargas, et. al, Cumulative link models for deep ordinal 
    classification, Neurocomputing, vol. 401, pp. 48–58, Aug 11 2020

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    and the CLM implementation from Vargas et. al
    https://github.com/ayrna/deep-ordinal-clm
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
            model_save_directory=None
    ):
        super(InceptionTimeOrdinalClassifier, self).__init__(
            model_name=model_name, model_save_directory=model_save_directory
        )

        self.verbose = verbose

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

        gpu_opt = tf.compat.v1.GPUOptions(
            allow_growth = True,
        )

        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            gpu_options=gpu_opt,
        )

        # tf.config.run_functions_eagerly(True)
        keras_c.backend.set_session(tf.compat.v1.Session(config=config))

        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(1)(output_layer)
        output_layer = keras.layers.BatchNormalization()(output_layer)      
        output_layer = CLM(
            nb_classes,
            link_function=self.link_function
        )(output_layer)

        model = keras_c.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=self.optimizer_lr),
            metrics=["accuracy"]
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
            model = self.build_model(self.input_shape, self.nb_classes)
            if self.verbose:    
                model.summary()
            self.history = model.fit(  
                X,
                y_onehot,
                batch_size=self.batch_size,
                epochs=self.nb_epochs,
                verbose=self.verbose,
                callbacks=self.callbacks
            )
            self.models.append(model)

        self._is_fitted = True

        return self
