# -*- coding: utf-8 -*-
"""Benchmarking orchestration module."""
__all__ = ["Orchestrator"]
__author__ = ["viktorkaz", "mloning"]

import os
import sys
import sktime
import logging
import numpy as np
from joblib import dump
from sklearn.base import clone
from dataclasses import dataclass
from sktime.benchmarking.tasks import TSCTask, TSRTask

np.set_printoptions(threshold=np.inf) # We need this to write long arrays in condor task_params.txt file

log = logging.getLogger()
console = logging.StreamHandler()
log.addHandler(console)

@dataclass
class CondorParams:
    batch_name : str
    requirements : str # Ej. '(Machine == "server.com")'
    getenv : bool = True
    should_transfer_files : str = "NO"
    request_CPUs : int = 0
    request_GPUs : int = 0
    request_memory : str = "1G"

class Orchestrator:
    """Fit and predict one or more estimators on one or more datasets."""

    def __init__(self, submission_params, tasks, datasets, strategies, cv, results, dl=False):
        # validate datasets and tasks
        self._validate_tasks_and_datasets(tasks, datasets)
        self.tasks = tasks
        self.datasets = datasets

        self.condor_folder_path = "./condor_files/"
        self.condor_tmp_path = self.condor_folder_path + "tmp_condor/"

        self.submission_params = submission_params

        # validate strategies
        self._validate_strategy_names(strategies)
        self.strategies = strategies

        self.cv = cv
        self.results = results

        # attach cv iterator to results object
        self.results.cv = cv

        self.dl = dl

        # progress trackers
        self.n_strategies = len(strategies)
        self.n_datasets = len(datasets)
        self._strategy_counter = 0
        self._dataset_counter = 0

    def _get_params(
        self, 
        overwrite_fitted_strategies,
        verbose ,
        overwrite_predictions,
        predict_on_train,
        save_fitted_strategies
        ):

        """Construct Params needed to run execution with condor"""

        params = ""

        if not os.path.exists(self.condor_folder_path):
            os.system("mkdir " + self.condor_folder_path)
        if not os.path.exists(self.condor_tmp_path):
            os.system("mkdir " + self.condor_tmp_path)            

        results_file_path = self.condor_tmp_path + 'results.pickle'
        with open(results_file_path, 'wb') as file:
            dump(self.results, file)
            
        cv_file_path = self.condor_tmp_path + 'cv.pickle'
        with open(cv_file_path, 'wb') as file:
            dump(self.cv, file) 

        for task, dataset in zip(self.tasks, self.datasets):
            self._strategy_counter = 0
            self._dataset_counter += 1

            dataset_file_path = self.condor_tmp_path + dataset.name + '.pickle'
            with open(dataset_file_path, 'wb') as file:
                dump(dataset, file) 

            task_file_path = self.condor_tmp_path + "task_" + str(self._dataset_counter) + '.pickle'
            with open(task_file_path, 'wb') as file:
                dump(task, file) 
            
            for strategy in self.strategies:
                self._strategy_counter += 1  # update counter
                strategy = clone(strategy)
                strategy_file_path = self.condor_tmp_path + strategy.name + '.pickle'
                with open(strategy_file_path, 'wb') as file:
                    dump(strategy, file) 

                for fold in range(0, self.cv.get_n_splits()):
                    filepath = os.path.join(self.results.path, strategy.name, dataset.name)

                    # Skip if test and train predictions exists and don't want to overwrite...
                    filename_test = f"{strategy.name}_test_{fold}.csv"
                    filename_train = f"{strategy.name}_train_{fold}.csv"
                    full_file_path_test = os.path.join(filepath, filename_test)
                    full_file_path_train = os.path.join(filepath, filename_train)
                    if (
                        os.path.exists(full_file_path_test)
                        and (os.path.exists(full_file_path_train) or not predict_on_train)
                       and overwrite_predictions == False
                       and overwrite_fitted_strategies == False):
                        print(f"SKIPPING - {filepath} already exists")
                        continue
                    
                    params += (
                        dataset_file_path   + "," +
                        task_file_path      + "," +
                        strategy_file_path  + "," +
                        results_file_path   + "," +
                        cv_file_path        + "," +
                        str(fold) + "," +
                        str(overwrite_fitted_strategies) + "," +
                        str(verbose) + "," +
                        str(overwrite_predictions) + "," +
                        str(predict_on_train) + "," +
                        str(save_fitted_strategies) + "," +
                        dataset.name + "\n"
                    )
        params = params.rstrip('\n')
        return params

    def fit_predict(
        self,
        overwrite_predictions=False,
        predict_on_train=False,
        save_fitted_strategies=True,
        overwrite_fitted_strategies=False,
        verbose=False
    ):
        """Fit and predict."""
        # check that for fitted strategies overwrite option is only set when
        # save option is set
        if overwrite_fitted_strategies and not save_fitted_strategies:
            raise ValueError(
                f"Can only overwrite fitted strategies "
                f"if fitted strategies are saved, but found: "
                f"overwrite_fitted_strategies="
                f"{overwrite_fitted_strategies} and "
                f"save_fitted_strategies="
                f"{save_fitted_strategies}"
            )

        params = self._get_params(
            overwrite_fitted_strategies,
            verbose,
            overwrite_predictions,
            predict_on_train,
            save_fitted_strategies
        )
        with open(self.condor_tmp_path + "task_params.txt", 'w') as f:
            f.write(params)
            f.close()
            
        self._write_condor_task_sub(self.submission_params, self.dl)
        os.system("condor_submit " + self.condor_tmp_path + "task.sub")

    @staticmethod
    def _predict_proba_one(strategy, task, data, y_true, y_pred):
        """Predict strategy on one dataset."""
        # TODO always try to get probabilistic predictions first, compute
        #  deterministic predictions using
        #  argmax to avoid rerunning predictions, only if no predict_proba
        #  is available, run predict

        # if the task is classification and the strategies supports
        # probabilistic predictions,
        # get probabilistic predictions
        if isinstance(task, TSCTask) and hasattr(strategy, "predict_proba"):
            return strategy.predict_proba(data)

            # otherwise, return deterministic predictions in expected format
            # else:
            #     n_class_true = len(np.unique(y_true))
            #     n_class_pred = len(np.unique(y_pred))
            #     n_classes = np.maximum(n_class_pred, n_class_true)
            #     n_predictions = len(y_pred)
            #     y_proba = (n_predictions, n_classes)
            #     y_proba = np.zeros(y_proba)
            #     y_proba[:, np.array(y_pred, dtype=int)] = 1

        else:
            return None

    @staticmethod
    def _validate_strategy_names(strategies):
        """Validate strategy names."""
        # Check uniqueness of strategy names
        names = [strategy.name for strategy in strategies]
        if not len(names) == len(set(names)):
            raise ValueError(
                f"Names of provided strategies are not unique: " f"{names}"
            )

        # Check for conflicts with estimator kwargs
        all_params = []
        for strategy in strategies:
            params = list(strategy.get_params(deep=False).keys())
            all_params.extend(params)

        invalid_names = set(names).intersection(set(all_params))
        if invalid_names:
            raise ValueError(
                f"Strategy names conflict with constructor "
                f"arguments: {sorted(invalid_names)}"
            )

        # Check for conflicts with double-underscore convention
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                f"Estimator names must not contain __: got " f"{invalid_names}"
            )

    @staticmethod
    def _validate_tasks_and_datasets(tasks, datasets):
        """Validate tasks."""
        # check input types
        if not isinstance(datasets, list):
            raise ValueError(f"datasets must be a list, but found: {type(datasets)}")
        if not isinstance(tasks, list):
            raise ValueError(f"tasks must be a list, but found: {type(tasks)}")

        # check if there is one task for each dataset
        if len(tasks) != len(datasets):
            raise ValueError(
                "Inconsistent number of datasets and tasks, "
                "there must be one task for each dataset"
            )

        # check if task is either time series regression or classification,
        # other tasks not supported yet
        if not all(isinstance(task, (TSCTask, TSRTask)) for task in tasks):
            raise NotImplementedError(
                "Currently, only time series classification and time series "
                "regression tasks are supported"
            )

        # check if all tasks are of the same type
        if not all(isinstance(task, type(tasks[0])) for task in tasks):
            raise ValueError("Not all tasks are of the same type")

    def _print_progress(
        self,
        dataset_name,
        strategy_name,
        cv_fold,
        train_or_test,
        fit_or_predict,
        verbose,
    ):
        """Print progress."""
        if verbose:
            fit_or_predict = fit_or_predict.capitalize()
            if train_or_test == "train" and fit_or_predict == "predict":
                on_train = " (training set)"
            else:
                on_train = ""

            log.warn(
                f"strategy: {self._strategy_counter}/{self.n_strategies} - "
                f"{strategy_name} "
                f"on CV-fold: {cv_fold} "
                f"of dataset: {self._dataset_counter}/{self.n_datasets} - "
                f"{dataset_name}{on_train}"
            )

    def _write_condor_task_sub(self, condor_params, dl):

        if not os.path.exists(self.condor_folder_path):
            os.system("mkdir " + self.condor_folder_path)
        output_path = self.condor_folder_path + "condor_output/"
        if not os.path.exists(output_path):
            os.system("mkdir " + output_path)

        if dl == True:
            executable_loc = sktime.__path__[0] + "/benchmarking/thread_dl.py"
        else:
            executable_loc = sktime.__path__[0] + "/benchmarking/thread.py"

        # get python environment path
        python_path = sys.executable

        with open(self.condor_tmp_path + 'task.sub', 'w') as f:
                file_str = ( f"""
                    batch_name \t = {condor_params.batch_name}
                    executable \t  = {python_path}
                    arguments \t  = {executable_loc} $(dataset_path) $(task_path) $(strategy_path) $(results_path) $(fold) $(train_idx) $(test_idx) $(overwrite_strats) $(verbose) $(overwrite_preds) $(predict_on_train) $(save_fitted_strategies) 
                    getenv \t  = {str(condor_params.getenv)}   
                    output \t  =   {output_path}out.out   
                    error \t  =   {output_path}error.err   
                    log \t  =   /dev/null   
                    should_transfer_files \t  =   {condor_params.should_transfer_files}   
                    request_CPUs \t  =   {str(condor_params.request_CPUs)}   
                    request_GPUs \t  =   {str(condor_params.request_GPUs)}   
                    request_memory \t  =   {condor_params.request_memory}   
                    requirements \t  =   {condor_params.requirements}   
                    queue dataset_path, task_path, strategy_path, results_path, fold, train_idx, test_idx, overwrite_strats, verbose, overwrite_preds, predict_on_train, save_fitted_strategies from {self.condor_tmp_path}task_params.txt
                """
                )
                f.write(file_str)
                f.close()