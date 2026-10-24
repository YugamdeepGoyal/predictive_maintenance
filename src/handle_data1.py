from pathlib import Path

import pandas as pd


class CMAPSSDataLoader:
    """Loads one subset of the NASA C-MAPSS turbofan dataset (default: FD001).

    Returns three DataFrames:
        train_df  - full run-to-failure history, with a raw (unclipped) RUL column
        test_df   - truncated histories of the test engines
        test_last - last recorded cycle of every test engine, with its true RUL
    """

    INDEX_COLS = ["engine_id", "cycle"]
    SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
    ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

    def __init__(self, data_dir=None, subset="FD001"):
        self.subset = subset
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "CMaps"
        self.data_dir = Path(data_dir)

    # ---------- file reading ----------
    def _read(self, path):
        # space-separated, no header
        return pd.read_csv(path, sep=r"\s+", header=None, names=self.ALL_COLS)

    def _read_rul(self, path):
        return pd.read_csv(path, header=None, names=["RUL"])

    # ---------- RUL construction ----------
    @staticmethod
    def _add_train_rul(train_df):
        # raw (unclipped) RUL = last cycle of the engine - current cycle
        max_cycle = train_df.groupby("engine_id")["cycle"].transform("max")
        train_df["RUL"] = max_cycle - train_df["cycle"]
        return train_df

    @staticmethod
    def _build_test_last(test_df, rul_true):
        # last recorded cycle of each engine, with its true RUL attached
        test_last = test_df.groupby("engine_id").last().reset_index()
        test_last["RUL"] = rul_true["RUL"].values
        return test_last

    # ---------- public API ----------
    def load(self):
        train_df = self._read(self.data_dir / f"train_{self.subset}.txt")
        test_df = self._read(self.data_dir / f"test_{self.subset}.txt")
        rul_true = self._read_rul(self.data_dir / f"RUL_{self.subset}.txt")

        train_df = self._add_train_rul(train_df)
        test_last = self._build_test_last(test_df, rul_true)
        return train_df, test_df, test_last


def load_fd001():
    """Backward-compatible helper so old `from src.handle_data import load_fd001` still works."""
    return CMAPSSDataLoader(subset="FD001").load()
