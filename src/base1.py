"""Shared abstract layer for the RUL project.

Pillars shown here:
  * Abstraction    - BaseRULModel / BaseRULPreprocessor are ABCs that define a contract.
  * Encapsulation  - private state (_model, _rul_cap, ...) exposed only via properties/validation.
  * Inheritance    - concrete models and preprocessors (in the notebooks) extend these bases.
  * Polymorphism   - ExperimentRunner works with ANY model/preprocessor that follows the contract.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, root_mean_squared_error)


class RegressionEvaluator:
    """Computes regression metrics and the NASA asymmetric scoring function."""

    @staticmethod
    def metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    @staticmethod
    def nasa_score(y_true, y_pred):
        errors = np.asarray(y_pred) - np.asarray(y_true)
        scores = np.where(errors < 0, np.exp(-errors / 13) - 1, np.exp(errors / 10) - 1)
        return scores.sum()


# ======================================================================
# Preprocessing contract
# ======================================================================
class BaseRULPreprocessor(ABC):
    """Abstract preprocessor: shared cleaning logic + an abstract `prepare()` step."""

    CONSTANT_COLS = ["setting_3", "sensor_1", "sensor_10", "sensor_18",
                     "sensor_19", "sensor_5", "sensor_16"]
    SENSOR_COLS = ["sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
                   "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
                   "sensor_15", "sensor_17", "sensor_20", "sensor_21"]

    def __init__(self, rul_cap=125, window=27, clip_test_target=True):
        self.rul_cap = rul_cap            # goes through the validating setter
        self.window = window
        self._clip_test_target = clip_test_target

    # ---- encapsulated, validated attributes ----
    @property
    def rul_cap(self):
        return self._rul_cap

    @rul_cap.setter
    def rul_cap(self, value):
        if value <= 0:
            raise ValueError("rul_cap must be positive")
        self._rul_cap = value

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        if int(value) < 1:
            raise ValueError("window must be >= 1")
        self._window = int(value)

    # ---- shared behaviour inherited by every subclass ----
    @staticmethod
    def find_constant_columns(df):
        return [c for c in df.columns if df[c].nunique() == 1]

    def _clip(self, series):
        return series.clip(upper=self._rul_cap)

    def _clean(self, train_df, test_df):
        """Drop constant columns, clip the training RUL, sort by engine/cycle."""
        train_df = train_df.drop(columns=self.CONSTANT_COLS).copy()
        test_df = test_df.drop(columns=self.CONSTANT_COLS).copy()
        train_df["RUL"] = self._clip(train_df["RUL"])
        train_df = train_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        test_df = test_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        return train_df, test_df

    def _test_target(self, test_last):
        y = test_last.sort_values("engine_id")["RUL"]
        if self._clip_test_target:
            y = self._clip(y)
        return y.to_numpy()

    @abstractmethod
    def prepare(self, train_df, test_df, test_last):
        """Return (X_train, y_train, X_test, y_test) in the form the model needs."""

    def __repr__(self):
        return f"{type(self).__name__}(rul_cap={self._rul_cap}, window={self._window})"


# ======================================================================
# Model contract
# ======================================================================
class BaseRULModel(ABC):
    """Abstract RUL model. Subclasses implement the model-specific steps;
    fit / predict / evaluate are written once here (template-method pattern)."""

    def __init__(self, name, rul_cap=125):
        self._name = name
        self.rul_cap = rul_cap
        self._model = None
        self._is_fitted = False

    # ---- encapsulation: read-only / validated access ----
    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def rul_cap(self):
        return self._rul_cap

    @rul_cap.setter
    def rul_cap(self, value):
        if value <= 0:
            raise ValueError("rul_cap must be positive")
        self._rul_cap = value

    # ---- abstract steps every model must implement ----
    @abstractmethod
    def _build_model(self, X):
        """Create and return the underlying estimator (X is given for shape information)."""

    @abstractmethod
    def _fit_model(self, X, y, **kwargs):
        """Train the underlying estimator."""

    @abstractmethod
    def _predict_raw(self, X):
        """Raw predictions from the underlying estimator."""

    @abstractmethod
    def save(self, path):
        """Persist the trained model."""

    # ---- optional hook (subclasses may override) ----
    def _postprocess(self, predictions):
        return predictions

    # ---- concrete public API (shared by all subclasses) ----
    def fit(self, X, y, **kwargs):
        self._model = self._build_model(X)
        self._fit_model(X, y, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError(f"{self._name} model is not fitted yet")
        return self._postprocess(np.asarray(self._predict_raw(X)).ravel())

    def evaluate(self, X, y_true):
        return RegressionEvaluator.metrics(y_true, self.predict(X))

    def __repr__(self):
        state = "fitted" if self._is_fitted else "not fitted"
        return f"{type(self).__name__}(name='{self._name}', {state})"


# ======================================================================
# Polymorphic orchestration
# ======================================================================
class ExperimentRunner:
    """Runs preprocessing -> training -> evaluation for ANY (preprocessor, model) pair.

    It only talks to the abstract interfaces, so it never needs to know whether it
    is driving XGBoost on flat rows or an LSTM on sequences (polymorphism)."""

    def __init__(self, preprocessor, model):
        if not isinstance(preprocessor, BaseRULPreprocessor):
            raise TypeError("preprocessor must extend BaseRULPreprocessor")
        if not isinstance(model, BaseRULModel):
            raise TypeError("model must extend BaseRULModel")
        self._preprocessor = preprocessor
        self._model = model
        self._data = {}
        self._results = None

    def run(self, train_df, test_df, test_last, **fit_kwargs):
        X_train, y_train, X_test, y_test = self._preprocessor.prepare(train_df, test_df, test_last)
        self._model.fit(X_train, y_train, **fit_kwargs)       # same call, different behaviour
        y_train_pred = self._model.predict(X_train)
        y_test_pred = self._model.predict(X_test)
        self._data = {"X_train": X_train, "y_train": np.asarray(y_train), "y_train_pred": y_train_pred,
                      "X_test": X_test, "y_test": np.asarray(y_test), "y_test_pred": y_test_pred}
        self._results = pd.DataFrame({
            "Training": RegressionEvaluator.metrics(self._data["y_train"], y_train_pred),
            "Testing": RegressionEvaluator.metrics(self._data["y_test"], y_test_pred),
        }).T
        return self._results

    @property
    def model(self):
        return self._model

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def results(self):
        return self._results

    @property
    def data(self):
        """Dictionary with X/y/prediction arrays of the last run (read-only copy)."""
        return dict(self._data)

    def nasa_score(self):
        return RegressionEvaluator.nasa_score(self._data["y_test"], self._data["y_test_pred"])

    def __repr__(self):
        return f"ExperimentRunner({self._preprocessor!r}, {self._model!r})"
