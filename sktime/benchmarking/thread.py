import sys

####### QUITAR ####### ####### #######
sys.path.append("/home/rayllon/GitHub/FORKED/sktime")
####### QUITAR ####### ####### #######

import logging
from pandas import Series, concat, Timestamp
from joblib import load

from sktime.benchmarking.condor_orchestration import Orchestrator


console = logging.StreamHandler()
log = logging.getLogger()
log.addHandler(console)

# Read console arguments
dataset_path = str(sys.argv[1]).strip()
target_path = str(sys.argv[2]).strip()
strategy_path = str(sys.argv[3]).strip()
results_path = str(sys.argv[4]).strip()
cv_path = str(sys.argv[5]).strip()
fold = int(sys.argv[6].strip())
overwrite_strategies = eval(sys.argv[7])
verbose = eval(sys.argv[8])
overwrite_predictions = eval(sys.argv[9])
predict_on_train = eval(sys.argv[10])
save_fitted_strategies = eval(sys.argv[11])
dataset_name = str(sys.argv[12]).strip()

with open(dataset_path, 'rb') as dataset_binary:
    dataset = load(dataset_binary)

with open(target_path, 'rb') as task_binary:
    task = load(task_binary)

with open(strategy_path, 'rb') as strategy_binary:
    strategy = load(strategy_binary)

with open(results_path, 'rb') as results_binary:
    results = load(results_binary)

with open(cv_path, 'rb') as cv_binary:
    cv = load(cv_binary)

# Do the work that standard condor.fit_predict() does in every for iteration
X_train, y_train, X_test, y_test = dataset.load_crude()

# check which results already exist
train_pred_exist = results.check_predictions_exist(
    strategy.name, dataset.name, fold, train_or_test="train"
)
test_pred_exist = results.check_predictions_exist(
    strategy.name, dataset.name, fold, train_or_test="test"
)
fitted_stategy_exists = results.check_fitted_strategy_exists(
    strategy.name, dataset.name, fold
)

if (
    not overwrite_predictions
    and not overwrite_strategies
    and test_pred_exist
    and (train_pred_exist or not predict_on_train)
    and (fitted_stategy_exists or not save_fitted_strategies)
):
    log.warning(
        f"Skipping strategy: {strategy.name} on CV-fold: "
        f"{fold} of dataset: {dataset.name}"
    )
    exit()

# Apply stratified resample
X_train, y_train, X_test, y_test = cv.apply(X_train, y_train, X_test, y_test, fold)

del cv

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

train = concat([X_train, Series(y_train)], axis=1)
test = concat([X_test, Series(y_test)], axis=1)

del X_train, y_train, X_test, y_test

# Combine train and test into single dataframe

train.rename(
    columns={train.columns[-1]: task.target}, inplace=True
)
test.rename(
    columns={test.columns[-1]: task.target}, inplace=True
)

if fitted_stategy_exists and not overwrite_strategies:
    log.warning("Fitted strategy already exists, it will be loaded instead of refitting")
    fit_estimator_start_time = Timestamp.now()
    strategy = results.load_fitted_strategy(strategy.name, dataset.name, fold)
    fit_estimator_end_time = Timestamp.now()
else:
    fit_estimator_start_time = Timestamp.now()
    strategy.fit(task, train)
    fit_estimator_end_time = Timestamp.now()

    if save_fitted_strategies:
        try:
            results.save_fitted_strategy(
                strategy, dataset_name=dataset.name, cv_fold=fold
            )
        except Exception as e:
            log.warning("ERROR SAVING STRATEGY!")
            log.warning(
                f"SAVING STRATEGY ERROR - "
                f"Strategy: {strategy.name} - "
                f"Dataset: {dataset.name}\n"
                f"* EXCEPTION: \n{e}"
            )

# optionally, predict on training set if predict on train is set
# to True and and overwrite is set to True
# or the predicted values do not already exist
if predict_on_train and (not train_pred_exist or overwrite_predictions):
    y_true = train.loc[:, task.target]
    predict_estimator_start_time = Timestamp.now()
    y_pred = strategy.predict(train)
    predict_estimator_end_time = Timestamp.now()
    train_idx = train.index

    y_proba = Orchestrator._predict_proba_one(strategy, task, train, y_true, y_pred)
    results.save_predictions(
        strategy_name=strategy.name,
        dataset_name=dataset.name,
        index=train_idx,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        cv_fold=fold,
        fit_estimator_start_time=fit_estimator_start_time,
        fit_estimator_end_time=fit_estimator_end_time,
        predict_estimator_start_time=predict_estimator_start_time,
        predict_estimator_end_time=predict_estimator_end_time,
        train_or_test="train",
    )

del train

if (not test_pred_exist or overwrite_predictions):
    y_true = test.loc[:, task.target]
    predict_estimator_start_time = Timestamp.now()
    y_pred = strategy.predict(test)
    predict_estimator_end_time = Timestamp.now()
    test_idx = test.index

    y_proba = Orchestrator._predict_proba_one(strategy, task, test, y_true, y_pred)

    del test

    try:
        results.save_predictions(
            dataset_name=dataset.name,
            strategy_name=strategy.name,
            index=test_idx,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            cv_fold=fold,
            fit_estimator_start_time=fit_estimator_start_time,
            fit_estimator_end_time=fit_estimator_end_time,
            predict_estimator_start_time=predict_estimator_start_time,
            predict_estimator_end_time=predict_estimator_end_time,
            train_or_test="test",
        )
        results.save()
        log.warning(
            f"Done! - Strat: {strategy.name} ·· Dataset: {dataset.name} ·· Fold: {fold} saved!"
        )
    except Exception as e:
        log.warning(
            f"SAVING PREDICTIONS ERROR - "
            f"Strategy: {strategy.name} - "
            f"Dataset: {dataset.name}\n"
            f"* EXCEPTION: \n{e}"
        )

del dataset, task, strategy, results