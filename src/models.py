"""Concrete preprocessors and models for the RUL project.

These used to live inside the notebooks. They sit here now so that both the
notebooks and the Streamlit app (app.py) can import the same classes.
TensorFlow and XGBoost are imported lazily, so each one is only needed when
you actually use the model that depends on it.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.base import BaseRULModel, BaseRULPreprocessor


# Tabular (XGBoost) side
class TabularPreprocessor(BaseRULPreprocessor):
    """Flat rows plus rolling window features, good for tree models."""

    def __init__(self, rul_cap=125, window=27):
        super().__init__(rul_cap=rul_cap, window=window, clip_test_target=True)

    def _add_rolling_features(self, df):
        df = df.sort_values(["engine_id", "cycle"]).copy()
        grouped = df.groupby("engine_id", sort=False)
        for col in self.SENSOR_COLS:
            df[f"{col}_roll_mean"] = grouped[col].transform(
                lambda s: s.rolling(window=self.window, min_periods=1).mean())
            df[f"{col}_roll_std"] = grouped[col].transform(
                lambda s: s.rolling(window=self.window, min_periods=2).std())
        return df

    def prepare(self, train_df, test_df, test_last):
        train_df, test_df = self._clean(train_df, test_df)
        train_df = self._add_rolling_features(train_df).drop(columns=["engine_id"])
        test_df = self._add_rolling_features(test_df)

        X_train = train_df.drop(columns=["RUL"])
        y_train = train_df["RUL"]
        # one row per test engine, its last recorded cycle
        X_test = test_df.groupby("engine_id").tail(1).drop(columns=["engine_id"])
        y_test = self._test_target(test_last)
        return X_train, y_train, X_test, y_test

    def engine_inputs(self, engine_df):
        df = engine_df.drop(columns=self.CONSTANT_COLS, errors="ignore")
        df = self._add_rolling_features(df)
        cycles = df["cycle"].to_numpy()
        return df.drop(columns=["engine_id", "RUL"], errors="ignore"), cycles


class XGBRULModel(BaseRULModel):
    """XGBoost regressor, predictions are clipped to the valid RUL range."""

    def __init__(self, rul_cap=125, random_state=42):
        super().__init__(name="XGBoost", rul_cap=rul_cap)
        self._random_state = random_state

    def _build_model(self, X):
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=self._random_state)

    def _fit_model(self, X, y, **kwargs):
        self._model.fit(X, y)

    def _predict_raw(self, X):
        return self._model.predict(X)

    def _postprocess(self, predictions):
        return np.clip(predictions, 0, self.rul_cap)

    def save(self, path="xgb_model.json"):
        self._model.save_model(path)

    def load(self, path="xgb_model.json"):
        from xgboost import XGBRegressor
        self._model = XGBRegressor()
        self._model.load_model(path)
        self._is_fitted = True
        return self


# Sequence (LSTM) side
class RollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds per engine rolling mean and std columns for the given sensors."""

    def __init__(self, sensor_cols, window=27):
        self.sensor_cols = sensor_cols
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.sort_values(["engine_id", "cycle"]).copy()
        grouped = df.groupby("engine_id", sort=False)
        for col in self.sensor_cols:
            df[f"{col}_roll_mean"] = grouped[col].transform(
                lambda s: s.rolling(window=self.window, min_periods=1).mean())
            df[f"{col}_roll_std"] = grouped[col].transform(
                lambda s: s.rolling(window=self.window, min_periods=2).std())
        return df


class IdColumnDropper(BaseEstimator, TransformerMixin):
    """Removes the engine_id column before scaling."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=["engine_id"])


class SequenceBuilder:
    """Cuts scaled rows into fixed length sliding windows, one engine at a time."""

    def __init__(self, seq_length=30):
        self.seq_length = seq_length

    def build_train(self, df, X, y):
        X, y = np.asarray(X), np.asarray(y)
        engine_ids = df["engine_id"].to_numpy()
        sequences, targets = [], []
        for engine_id in df["engine_id"].unique():
            indices = np.flatnonzero(engine_ids == engine_id)
            engine_X = X[indices]
            if len(engine_X) < self.seq_length:
                continue
            for end in range(self.seq_length, len(engine_X) + 1):
                sequences.append(engine_X[end - self.seq_length:end])
                targets.append(y[indices[end - 1]])
        return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def build_test(self, df, X):
        """One window per test engine, its last seq_length cycles."""
        X = np.asarray(X)
        engine_ids = df["engine_id"].to_numpy()
        sequences = []
        for engine_id in df["engine_id"].unique():
            engine_X = X[np.flatnonzero(engine_ids == engine_id)]
            if len(engine_X) >= self.seq_length:
                sequences.append(engine_X[-self.seq_length:])
        return np.asarray(sequences, dtype=np.float32)

    def build_all_windows(self, X):
        """Every window ending at cycle seq_length, seq_length + 1, ... for ONE engine."""
        X = np.asarray(X)
        if len(X) < self.seq_length:
            return np.empty((0, self.seq_length, X.shape[1]), dtype=np.float32)
        windows = [X[end - self.seq_length:end] for end in range(self.seq_length, len(X) + 1)]
        return np.asarray(windows, dtype=np.float32)


class SequencePreprocessor(BaseRULPreprocessor):
    """Scaled windows shaped (samples, seq_length, features) for recurrent models."""

    def __init__(self, rul_cap=125, window=27, seq_length=30):
        super().__init__(rul_cap=rul_cap, window=window, clip_test_target=False)
        self._seq_length = seq_length
        self._builder = SequenceBuilder(seq_length)
        self._fitted = False
        self._pipeline = Pipeline([
            ("rolling_features", RollingFeatureTransformer(self.SENSOR_COLS, self.window)),
            ("drop_id", IdColumnDropper()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ])

    @property
    def seq_length(self):
        return self._seq_length

    def prepare(self, train_df, test_df, test_last):
        train_df, test_df = self._clean(train_df, test_df)

        X_train_df = train_df.drop(columns=["RUL"])
        y_train = train_df["RUL"].to_numpy(dtype=np.float32)

        # the scaler only ever sees training data
        X_train_scaled = self._pipeline.fit_transform(X_train_df)
        self._fitted = True
        X_test_scaled = self._pipeline.transform(test_df)

        X_train_seq, y_train_seq = self._builder.build_train(X_train_df, X_train_scaled, y_train)
        X_test_seq = self._builder.build_test(test_df, X_test_scaled)
        y_test = self._test_target(test_last)
        return X_train_seq, y_train_seq, X_test_seq, y_test

    def engine_inputs(self, engine_df):
        if not self._fitted:
            raise RuntimeError("Call prepare() once first so the scaler is fitted")
        df = engine_df.drop(columns=self.CONSTANT_COLS + ["RUL"], errors="ignore")
        df = df.sort_values(["engine_id", "cycle"])
        scaled = self._pipeline.transform(df)
        windows = self._builder.build_all_windows(scaled)
        cycles = df["cycle"].to_numpy()[self._seq_length - 1:]
        return windows, cycles


class LSTMRULModel(BaseRULModel):
    """Single layer LSTM regressor for remaining useful life."""

    def __init__(self, rul_cap=125, seq_length=30, units=64, learning_rate=1e-3):
        super().__init__(name="LSTM", rul_cap=rul_cap)
        self._seq_length = seq_length
        self._units = units
        self._learning_rate = learning_rate
        self.__history = None          # name mangled on purpose, truly private

    @property
    def history(self):
        return self.__history

    def _build_model(self, X):
        from tensorflow import keras
        from tensorflow.keras.layers import Dense, LSTM, Input
        from tensorflow.keras.models import Sequential

        n_features = X.shape[2]
        model = Sequential([
            Input(shape=(self._seq_length, n_features)),
            LSTM(self._units),
            Dense(1, activation="linear"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self._learning_rate),
            loss="mse",
            metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                     keras.metrics.R2Score(name="r2_score")],
        )
        return model

    def _fit_model(self, X, y, validation_split=0.15, epochs=100, batch_size=128, **kwargs):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        ]
        self.__history = self._model.fit(X, y, validation_split=validation_split, epochs=epochs,
                                         batch_size=batch_size, callbacks=callbacks)

    def _predict_raw(self, X):
        return self._model.predict(X, verbose=0)

    def summary(self):
        self._model.summary()

    def save(self, path="lstm_model.keras"):
        self._model.save(path)

    def load(self, path="lstm_model.keras"):
        from tensorflow import keras
        self._model = keras.models.load_model(path, compile=False)
        self._is_fitted = True
        return self
