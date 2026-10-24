"""Engine Health Console

A Streamlit front end for the C-MAPSS FD001 remaining useful life models.
Run it from the project root (the folder that has src/ and CMaps/):

    streamlit run app.py

It expects lstm_model.keras and xgb_model.json next to this file. Both are
written by the notebooks. If xgb_model.json is missing the app trains and
saves one itself, if lstm_model.keras is missing the app runs XGBoost only.
"""
import random
import time
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.base import BaseRULModel, BaseRULPreprocessor, RegressionEvaluator
from src.handle_data import CMAPSSDataLoader
from src.models import LSTMRULModel, SequencePreprocessor, TabularPreprocessor, XGBRULModel

ROOT = Path(__file__).resolve().parent
XGB_PATH = ROOT / "xgb_model.json"
LSTM_PATH = ROOT / "lstm_model.keras"

RUL_CAP = 125
SENSORS = BaseRULPreprocessor.SENSOR_COLS
COLORS = {"Actual": "#8a8f98", "XGBoost": "#e8833a", "LSTM": "#3b82c4"}

PAGE_STYLE = """
<style>
.hero {
    padding: 1.1rem 1.4rem;
    border-radius: 14px;
    background: linear-gradient(120deg, rgba(59,130,196,0.18), rgba(232,131,58,0.14));
    margin-bottom: 0.8rem;
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: 0.3rem 0 0 0; opacity: 0.8; }
.badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    color: white;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
</style>
"""


def status_for(rul, critical, watch):
    if rul <= critical:
        return "CRITICAL", "#d64545"
    if rul <= watch:
        return "WATCH", "#e0a526"
    return "HEALTHY", "#2e9e5b"


def badge(rul, critical, watch):
    text, color = status_for(rul, critical, watch)
    return f'<span class="badge" style="background:{color}">{text}</span>'


# Data and model plumbing
class ModelBundle:
    """A model together with the preprocessor that knows how to feed it.

    Everything the UI does goes through the two abstract interfaces, so this
    class never has to ask whether it is holding XGBoost or an LSTM."""

    def __init__(self, model, preprocessor, fleet_inputs):
        self.model = model
        self.preprocessor = preprocessor
        self.label = model.name
        self.fleet_pred = np.clip(model.predict(fleet_inputs), 0, RUL_CAP)

    def history(self, engine_df):
        """Predicted RUL at every cycle where the model has enough context."""
        inputs, cycles = self.preprocessor.engine_inputs(engine_df)
        preds = np.clip(self.model.predict(inputs), 0, RUL_CAP)
        return pd.DataFrame({"cycle": cycles, "RUL": preds})


class Assets:
    """Loads the dataset once and prepares every model that is available."""

    def __init__(self):
        loader = CMAPSSDataLoader(subset="FD001")
        self.train_df, self.test_df, test_last = loader.load()
        self.test_last = test_last.sort_values("engine_id").reset_index(drop=True)

        self.engine_ids = self.test_last["engine_id"].tolist()
        self.final_rul = dict(zip(self.test_last["engine_id"], self.test_last["RUL"]))
        self.true_final = np.minimum(self.test_last["RUL"].to_numpy(), RUL_CAP)

        self.sensor_mean = self.train_df[SENSORS].mean()
        self.sensor_std = self.train_df[SENSORS].std()

        self.notes = []
        self.bundles = []
        self._add_xgboost()
        self._add_lstm()

    def _add_xgboost(self):
        prep = TabularPreprocessor(rul_cap=RUL_CAP)
        X_train, y_train, X_test, _ = prep.prepare(self.train_df, self.test_df, self.test_last)
        model = XGBRULModel(rul_cap=RUL_CAP)
        if XGB_PATH.exists():
            model.load(str(XGB_PATH))
        else:
            model.fit(X_train, y_train)
            model.save(str(XGB_PATH))
            self.notes.append("xgb_model.json was missing, so I trained XGBoost just now and saved it.")
        self.bundles.append(ModelBundle(model, prep, X_test))

    def _add_lstm(self):
        if not LSTM_PATH.exists():
            self.notes.append("lstm_model.keras not found. Run lstm_model.ipynb to create it "
                              "and the LSTM will show up here.")
            return
        try:
            prep = SequencePreprocessor(rul_cap=RUL_CAP)
            _, _, X_test, _ = prep.prepare(self.train_df, self.test_df, self.test_last)
            model = LSTMRULModel(rul_cap=RUL_CAP).load(str(LSTM_PATH))
            self.bundles.append(ModelBundle(model, prep, X_test))
        except Exception as err:
            self.notes.append(f"The LSTM could not be loaded: {err}")

    # small helpers used by the tabs
    def bundle(self, label):
        return next(b for b in self.bundles if b.label == label)

    def engine_frame(self, engine_id):
        df = self.test_df[self.test_df["engine_id"] == engine_id].sort_values("cycle")
        return df.reset_index(drop=True)

    def true_rul(self, engine_id, cycles):
        """Actual RUL at each cycle. The test files only reveal it at the last
        cycle, but it drops by one every cycle, so we can walk backwards."""
        cycles = np.asarray(cycles)
        last_cycle = self.test_df.loc[self.test_df["engine_id"] == engine_id, "cycle"].max()
        return np.minimum(self.final_rul[engine_id] + (last_cycle - cycles), RUL_CAP)

    def inject(self, engine_df, sensor, sigma, n_last):
        """Return a copy where one sensor drifts away from normal over the last n cycles."""
        out = engine_df.copy()
        n_last = min(n_last, len(out))
        tail = out.index[-n_last:]
        ramp = np.linspace(1 / n_last, 1.0, n_last)
        out.loc[tail, sensor] = out.loc[tail, sensor] + ramp * sigma * self.sensor_std[sensor]
        return out


@st.cache_resource(show_spinner="Loading data and models. The first run takes a little while.")
def get_assets():
    return Assets()


@st.cache_data(show_spinner=False)
def cached_history(_assets, label, engine_id):
    # _assets starts with an underscore so streamlit skips hashing it
    return _assets.bundle(label).history(_assets.engine_frame(engine_id))


@dataclass
class Settings:
    bundles: list
    engine: int
    critical: int
    watch: int


# The app
class ConsoleApp:
    def __init__(self, assets):
        self.assets = assets
        self.labels = [b.label for b in assets.bundles]

    def _pick_random_engine(self):
        st.session_state["engine"] = random.choice(self.assets.engine_ids)

    def _sidebar(self):
        st.sidebar.header("Controls")
        chosen = st.sidebar.multiselect("Models", self.labels, default=self.labels)

        if "engine" not in st.session_state:
            st.session_state["engine"] = self.assets.engine_ids[0]
        st.sidebar.selectbox("Engine", self.assets.engine_ids, key="engine")
        st.sidebar.button("Pick a random engine", on_click=self._pick_random_engine)

        st.sidebar.subheader("Alert levels")
        critical = st.sidebar.slider("Critical when RUL is below", 5, 60, 30)
        watch = st.sidebar.slider("Watch when RUL is below", 20, 110, 70)
        watch = max(watch, critical + 1)

        for note in self.assets.notes:
            st.sidebar.info(note)

        bundles = [b for b in self.assets.bundles if b.label in chosen]
        return Settings(bundles, st.session_state["engine"], critical, watch)

    def run(self):
        st.markdown(PAGE_STYLE, unsafe_allow_html=True)
        st.markdown(
            '<div class="hero"><h1>Engine Health Console</h1>'
            "<p>Pick a real turbofan from the NASA C-MAPSS test set, replay its life cycle by cycle "
            "and watch the models guess how much life it has left. The true answer is hidden from "
            "them, so what you see is an honest test.</p></div>",
            unsafe_allow_html=True,
        )
        s = self._sidebar()
        if not s.bundles:
            st.warning("Choose at least one model in the sidebar.")
            return

        replay, fault, fleet, hood = st.tabs(
            ["Live replay", "Fault injection", "Fleet view", "Under the hood"])
        with replay:
            self._tab_replay(s)
        with fault:
            self._tab_fault(s)
        with fleet:
            self._tab_fleet(s)
        with hood:
            self._tab_hood(s)

    # Tab 1: replay
    def _tab_replay(self, s):
        eng = self.assets.engine_frame(s.engine)
        hist = {b.label: cached_history(self.assets, b.label, s.engine) for b in s.bundles}
        first_ok = int(max(h["cycle"].min() for h in hist.values()))
        last = int(eng["cycle"].max())
        if first_ok >= last:
            st.info("This engine has too little history for the selected models.")
            return

        watch_sensors = st.multiselect(
            "Sensors to watch", SENSORS, default=["sensor_11", "sensor_4", "sensor_12"])
        left, right = st.columns([5, 1])
        with left:
            cycle = st.slider("Cycle", first_ok, last, last, key=f"cycle_{s.engine}")
        with right:
            st.write("")
            play = st.button("Play replay")

        slot = st.empty()
        if play:
            step = max(1, (last - first_ok) // 70)
            for c in list(range(first_ok, last + 1, step)) + [last]:
                with slot.container():
                    self._draw_replay(s, eng, hist, watch_sensors, c, first_ok, last)
                time.sleep(0.05)
        else:
            with slot.container():
                self._draw_replay(s, eng, hist, watch_sensors, cycle, first_ok, last)

    def _draw_replay(self, s, eng, hist, watch_sensors, c, first_ok, last):
        a = self.assets
        actual_now = float(a.true_rul(s.engine, [c])[0])

        cols = st.columns(len(s.bundles) + 1)
        cols[0].metric("Actual RUL", f"{actual_now:.0f} cycles")
        cols[0].caption(f"Engine {s.engine}, cycle {c} of {last}")
        for col, b in zip(cols[1:], s.bundles):
            h = hist[b.label]
            pred = float(h.loc[h["cycle"] == c, "RUL"].iloc[0])
            col.metric(f"{b.label} estimate", f"{pred:.0f} cycles",
                       delta=f"{pred - actual_now:+.0f} vs actual", delta_color="off")
            col.markdown(badge(pred, s.critical, s.watch), unsafe_allow_html=True)
            col.progress(float(min(pred / RUL_CAP, 1.0)))

        # remaining life: the actual curve is drawn in full, predictions only up to now
        all_cycles = eng["cycle"].to_numpy()
        frames = [pd.DataFrame({"cycle": all_cycles, "RUL": a.true_rul(s.engine, all_cycles),
                                "series": "Actual"})]
        for b in s.bundles:
            h = hist[b.label]
            frames.append(h[h["cycle"] <= c].assign(series=b.label))
        data = pd.concat(frames)

        names = ["Actual"] + [b.label for b in s.bundles]
        color = alt.Color("series:N", legend=alt.Legend(title=None, orient="top"),
                          scale=alt.Scale(domain=names, range=[COLORS[n] for n in names]))
        x_scale = alt.Scale(domain=[first_ok, last])

        lines = alt.Chart(data).mark_line(strokeWidth=2.5).encode(
            x=alt.X("cycle:Q", title="Cycle", scale=x_scale),
            y=alt.Y("RUL:Q", title="Remaining useful life (cycles)",
                    scale=alt.Scale(domain=[0, RUL_CAP + 5])),
            color=color)
        critical_band = alt.Chart(pd.DataFrame({"lo": [0], "hi": [s.critical]})).mark_rect(
            opacity=0.12, color="#d64545").encode(y="lo:Q", y2="hi:Q")
        watch_band = alt.Chart(pd.DataFrame({"lo": [s.critical], "hi": [s.watch]})).mark_rect(
            opacity=0.10, color="#e0a526").encode(y="lo:Q", y2="hi:Q")
        now_line = alt.Chart(pd.DataFrame({"cycle": [c]})).mark_rule(
            color="#888888", strokeDash=[4, 4]).encode(x="cycle:Q")
        st.altair_chart((critical_band + watch_band + lines + now_line)
                        .properties(height=320, width="container"))

        if watch_sensors:
            window = eng.loc[eng["cycle"] <= c, ["cycle"] + watch_sensors].set_index("cycle")
            z = window.sub(a.sensor_mean[watch_sensors], axis="columns")
            z = z.div(a.sensor_std[watch_sensors], axis="columns")
            z = z.reset_index().melt("cycle", var_name="sensor", value_name="z")
            sensor_chart = alt.Chart(z).mark_line().encode(
                x=alt.X("cycle:Q", title="Cycle", scale=x_scale),
                y=alt.Y("z:Q", title="Std devs from fleet average"),
                color=alt.Color("sensor:N", legend=alt.Legend(title=None, orient="top")))
            st.altair_chart(sensor_chart.properties(height=200, width="container"))

    # Tab 2: fault injection
    def _tab_fault(self, s):
        st.write("Pick a sensor and make it drift away from normal over the last few cycles of "
                 "this engine. The models re read the doctored history and tell you how much "
                 "life they think just disappeared.")
        c1, c2, c3 = st.columns(3)
        sensor = c1.selectbox("Sensor", SENSORS, index=SENSORS.index("sensor_11"))
        sigma = c2.slider("Drift size (std devs)", -4.0, 4.0, 2.0, 0.5)
        n_last = c3.slider("Applied over the last N cycles", 3, 60, 15)

        eng = self.assets.engine_frame(s.engine)
        doctored = self.assets.inject(eng, sensor, sigma, n_last)

        cols = st.columns(len(s.bundles))
        for col, b in zip(cols, s.bundles):
            before = float(cached_history(self.assets, b.label, s.engine)["RUL"].iloc[-1])
            after = float(b.history(doctored)["RUL"].iloc[-1])
            col.metric(f"{b.label}: RUL now", f"{after:.0f} cycles",
                       delta=f"{after - before:+.1f} from {before:.0f}", delta_color="off")
            col.markdown(badge(after, s.critical, s.watch), unsafe_allow_html=True)

        trace = pd.concat([
            eng[["cycle", sensor]].rename(columns={sensor: "value"}).assign(version="Original"),
            doctored[["cycle", sensor]].rename(columns={sensor: "value"}).assign(version="Doctored"),
        ])
        trace = trace[trace["cycle"] >= trace["cycle"].max() - 80]
        chart = alt.Chart(trace).mark_line().encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("value:Q", title=sensor, scale=alt.Scale(zero=False)),
            color=alt.Color("version:N", legend=alt.Legend(title=None, orient="top"),
                            scale=alt.Scale(domain=["Original", "Doctored"],
                                            range=["#8a8f98", "#d64545"])))
        st.altair_chart(chart.properties(height=240, width="container"))

        with st.expander("Which sensor does this engine react to most?"):
            st.write("Runs the same drift on every informative sensor, one at a time, "
                     "and records how far the estimate moves.")
            pick = st.selectbox("Model", [b.label for b in s.bundles], key="sweep_model")
            if st.button("Run sweep"):
                bundle = self.assets.bundle(pick)
                base = float(cached_history(self.assets, pick, s.engine)["RUL"].iloc[-1])
                rows, bar = [], st.progress(0.0)
                for i, name in enumerate(SENSORS):
                    new = float(bundle.history(
                        self.assets.inject(eng, name, sigma, n_last))["RUL"].iloc[-1])
                    rows.append({"sensor": name, "change": new - base, "size": abs(new - base)})
                    bar.progress((i + 1) / len(SENSORS))
                bar.empty()
                res = pd.DataFrame(rows)
                sweep = alt.Chart(res).mark_bar().encode(
                    y=alt.Y("sensor:N", sort=alt.EncodingSortField(field="size", order="descending"),
                            title=None),
                    x=alt.X("change:Q", title="Change in predicted RUL (cycles)"),
                    color=alt.condition(alt.datum.change > 0, alt.value("#2e9e5b"),
                                        alt.value("#d64545")),
                    tooltip=["sensor", "change"])
                st.altair_chart(sweep.properties(height=380, width="container"))

    # Tab 3: fleet
    def _tab_fleet(self, s):
        a = self.assets
        truth = a.true_final

        score_rows = []
        for b in s.bundles:
            m = RegressionEvaluator.metrics(truth, b.fleet_pred)
            m["NASA score"] = RegressionEvaluator.nasa_score(truth, b.fleet_pred)
            score_rows.append(pd.Series(m, name=b.label))
        st.subheader("How well do the models do on all 100 test engines?")
        st.dataframe(pd.DataFrame(score_rows).round(3))
        st.caption("Scored against actual RUL capped at 125 for every model, so the comparison is "
                   "fair. Lower is better for everything except R2. The notebook scores the LSTM "
                   "against the uncapped value, so its numbers will differ a little.")

        pts = pd.concat([
            pd.DataFrame({"engine": a.engine_ids, "Actual": truth,
                          "Predicted": b.fleet_pred, "model": b.label})
            for b in s.bundles])
        names = [b.label for b in s.bundles]
        dots = alt.Chart(pts).mark_circle(size=70, opacity=0.75).encode(
            x=alt.X("Actual:Q", title="Actual RUL"),
            y=alt.Y("Predicted:Q", title="Predicted RUL"),
            color=alt.Color("model:N", legend=alt.Legend(title=None, orient="top"),
                            scale=alt.Scale(domain=names, range=[COLORS[n] for n in names])),
            tooltip=["engine", "model", "Actual", "Predicted"])
        diagonal = alt.Chart(pd.DataFrame({"v": [0, RUL_CAP]})).mark_line(
            color="#888888", strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
        st.altair_chart((diagonal + dots).properties(height=380, width="container"))
        st.caption("Dots on the dashed line are perfect predictions.")

        st.subheader("Maintenance queue")
        queue = pd.DataFrame({"Engine": a.engine_ids, "Actual RUL": truth})
        for b in s.bundles:
            queue[b.label] = b.fleet_pred
        worst = queue[[b.label for b in s.bundles]].min(axis=1)
        queue["Status"] = [status_for(v, s.critical, s.watch)[0] for v in worst]
        queue = queue.assign(_sort=worst).sort_values("_sort").drop(columns="_sort").head(15)
        config = {b.label: st.column_config.ProgressColumn(
            b.label, min_value=0, max_value=RUL_CAP, format="%d") for b in s.bundles}
        st.dataframe(queue, hide_index=True, column_config=config)
        st.caption("The 15 engines closest to failure according to the most pessimistic selected model.")

        counts = pd.DataFrame({
            b.label: {"Critical": int((b.fleet_pred <= s.critical).sum()),
                      "Watch": int(((b.fleet_pred > s.critical) & (b.fleet_pred <= s.watch)).sum()),
                      "Healthy": int((b.fleet_pred > s.watch).sum())}
            for b in s.bundles})
        st.write("Fleet status by model")
        st.dataframe(counts.T)

        if len(s.bundles) >= 2:
            first, second = s.bundles[0], s.bundles[1]
            gap = pd.DataFrame({"Engine": a.engine_ids, "Actual RUL": truth,
                                first.label: first.fleet_pred, second.label: second.fleet_pred})
            gap["Gap"] = (gap[first.label] - gap[second.label]).abs()
            st.write(f"Engines where {first.label} and {second.label} disagree the most")
            st.dataframe(gap.sort_values("Gap", ascending=False).head(5).round(1), hide_index=True)

    # Tab 4: OOP showcase
    def _tab_hood(self, s):
        st.write("Everything above runs through two abstract base classes. The UI only ever calls "
                 "`engine_inputs`, `fit`, `predict` and `load`, and each concrete class decides "
                 "what that means for itself.")

        st.subheader("Inheritance and abstraction")
        for b in self.assets.bundles:
            model_chain = " > ".join(c.__name__ for c in type(b.model).__mro__)
            prep_chain = " > ".join(c.__name__ for c in type(b.preprocessor).__mro__)
            st.code(f"{model_chain}\n{prep_chain}", language="text")

        st.subheader("Polymorphism: who overrides what")
        model_steps = ["_build_model", "_fit_model", "_predict_raw", "_postprocess",
                       "save", "load", "fit", "predict", "evaluate"]
        st.dataframe(self._override_table(BaseRULModel, [type(b.model) for b in self.assets.bundles],
                                          model_steps), hide_index=True)
        prep_steps = ["prepare", "engine_inputs", "_clean", "find_constant_columns"]
        st.dataframe(self._override_table(BaseRULPreprocessor,
                                          [type(b.preprocessor) for b in self.assets.bundles],
                                          prep_steps), hide_index=True)

        st.subheader("Encapsulation: try to break a model")
        st.write("`rul_cap` is private behind a property that validates the value. "
                 "Try something silly on a throwaway model.")
        value = st.number_input("Set rul_cap to", value=-5, step=5)
        scratch = XGBRULModel()
        try:
            scratch.rul_cap = value
            st.success(f"Accepted. rul_cap is now {scratch.rul_cap}.")
        except ValueError as err:
            st.error(f"Rejected by the setter: {err}")
        try:
            BaseRULModel("direct")
        except TypeError as err:
            st.caption(f"Abstraction check, building the base class directly fails: {err}")

    @staticmethod
    def _override_table(base, classes, steps):
        rows = []
        for step in steps:
            row = {"method": step,
                   base.__name__: "abstract" if step in base.__abstractmethods__ else "implemented"}
            for cls in classes:
                row[cls.__name__] = "overridden" if step in vars(cls) else "inherited"
            rows.append(row)
        return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Engine Health Console", layout="wide")
    ConsoleApp(get_assets()).run()


if __name__ == "__main__":
    main()
