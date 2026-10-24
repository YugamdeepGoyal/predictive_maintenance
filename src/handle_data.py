import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def _read(path):
    # space-separated, no header
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)


def load_fd001():
    BASE_DIR = Path(__file__).resolve().parent.parent / "CMaps"
    path = BASE_DIR / "train_FD001.txt"
    train_df = _read(path)
    test_df = _read(BASE_DIR / "test_FD001.txt")
    rul_true = pd.read_csv(BASE_DIR / "RUL_FD001.txt", header=None, names=["RUL"])
    # raw (unclipped) RUL for training = last cycle of the engine - current cycle
    max_cycle = train_df.groupby("engine_id")["cycle"].transform("max")
    train_df["RUL"] = max_cycle - train_df["cycle"]

    # test: last recorded cycle of each engine, with its true RUL attached
    test_last = test_df.groupby("engine_id").last().reset_index()
    test_last["RUL"] = rul_true["RUL"].values

    return train_df, test_df, test_last


