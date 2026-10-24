# """Engine Life Predictor

# A Streamlit front end for the C-MAPSS FD001 remaining useful life models.

#   * Predict            - the main page. Upload a CSV or type in sensor readings and get
#                          one clear answer: how many flight cycles the engine has left.
#   * Analyst dashboard  - the replay / fault injection / fleet / OOP view, kept as a
#                          second option for analysts and developers.

# Run it from the project root (the folder that has src/ and CMaps/):

#     streamlit run app.py

# It expects lstm_model.keras and xgb_model.json next to this file. Both are
# written by the notebooks. If xgb_model.json is missing the app trains and
# saves one itself, if lstm_model.keras is missing the app runs XGBoost only.
# """
# import base64
# import io
# import math
# import random
# import time
# from dataclasses import dataclass
# from pathlib import Path

# import altair as alt
# import numpy as np
# import pandas as pd
# import streamlit as st

# from src.base import BaseRULModel, BaseRULPreprocessor, RegressionEvaluator
# from src.handle_data import CMAPSSDataLoader
# from src.models import LSTMRULModel, SequencePreprocessor, TabularPreprocessor, XGBRULModel

# ROOT = Path(__file__).resolve().parent
# XGB_PATH = ROOT / "xgb_model.json"
# LSTM_PATH = ROOT / "lstm_model.keras"

# RUL_CAP = 125
# SENSORS = BaseRULPreprocessor.SENSOR_COLS
# COLORS = {"Actual": "#8a8f98", "XGBoost": "#e8833a", "LSTM": "#3b82c4"}

# ZONE = {"CRITICAL": "#E5594A", "WATCH": "#E9B23A", "HEALTHY": "#4FB27E"}

# PAGE_STYLE = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600&display=swap');

# /* type: Barlow for reading, Barlow Condensed for headings and big numbers.
#    Only text elements are targeted so Streamlit's icon fonts keep working. */
# .stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea {
#     font-family: 'Barlow', 'Segoe UI', Arial, sans-serif;
# }
# .stApp h1, .stApp h2, .stApp h3, [data-testid="stMetricValue"] {
#     font-family: 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif;
#     font-weight: 600;
# }
# .stApp h3 { font-size: 1.7rem; letter-spacing: 0.01em; }
# .block-container { padding-top: 2.2rem; }
# .stButton > button, .stDownloadButton > button { border-radius: 4px; }

# /* page heading */
# .hero {
#     display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
#     padding: 0.2rem 0 1.2rem 0; margin-bottom: 1.3rem;
#     border-bottom: 3px solid #17708A;
# }
# .stApp .hero h1 { font-size: 3.2rem; line-height: 1.05; margin: 0; padding: 0; max-width: 17ch; }
# .hero p { margin: 0.7rem 0 0 0; max-width: 54ch; opacity: 0.78; font-size: 1.05rem; }
# .hero .fan { width: 128px; height: 128px; flex: none; }
# @media (max-width: 640px) { .hero .fan { display: none; } .stApp .hero h1 { font-size: 2.4rem; } }

# /* the result, laid out like an instrument panel */
# .panel {
#     background: #1C2E3B; color: #EEF2F4; border-radius: 6px;
#     padding: 1.6rem 1.9rem 0 1.9rem; margin: 0.4rem 0 1rem 0;
# }
# .panel-top { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem 2.6rem; }
# .panel .gauge { width: 380px; max-width: 100%; height: auto; flex: 0 1 380px; }
# .panel .readout { flex: 1 1 240px; }
# .panel .small { color: #9FB0BC; font-size: 1rem; }
# .panel .num {
#     font-family: 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif;
#     font-size: 5.6rem; font-weight: 600; line-height: 1.02;
# }
# .panel .num span { font-size: 1.4rem; font-weight: 500; color: #9FB0BC; margin-left: 0.5rem; }
# .panel .pill {
#     display: inline-block; padding: 2px 12px; border-radius: 4px;
#     color: #10202B; font-weight: 600; font-size: 0.95rem;
# }
# .panel .advice { font-size: 1.15rem; margin: 0.8rem 0 0.3rem 0; }
# .panel .facts { display: flex; flex-wrap: wrap; margin-top: 1.4rem; border-top: 1px solid rgba(255,255,255,0.15); }
# .panel .facts div { flex: 1 1 150px; padding: 0.85rem 0 1.1rem 0; font-size: 1.05rem; }
# .panel .facts div + div { padding-left: 1.2rem; border-left: 1px solid rgba(255,255,255,0.15); }
# .panel .facts b { display: block; font-weight: 500; color: #9FB0BC; font-size: 0.85rem; }

# /* small status label used by the analyst dashboard */
# .badge {
#     display: inline-block; padding: 2px 11px; border-radius: 4px;
#     color: white; font-size: 0.85rem; font-weight: 600;
# }
# </style>
# """


# def status_for(rul, critical, watch):
#     if rul <= critical:
#         return "CRITICAL", "#d64545"
#     if rul <= watch:
#         return "WATCH", "#e0a526"
#     return "HEALTHY", "#2e9e5b"


# def badge(rul, critical, watch):
#     text, color = status_for(rul, critical, watch)
#     return f'<span class="badge" style="background:{color}">{text.capitalize()}</span>'


# def svg_img(svg, css_class="", alt=""):
#     """Embed an SVG as an <img>, which every Streamlit version renders reliably."""
#     uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
#     return f'<img class="{css_class}" src="{uri}" alt="{alt}">'


# def _polar(cx, cy, radius, value):
#     """Point on the dial: RUL 0 sits on the far left, RUL_CAP on the far right."""
#     angle = math.pi * (1 - min(max(value, 0), RUL_CAP) / RUL_CAP)
#     return cx + radius * math.cos(angle), cy - radius * math.sin(angle)


# def gauge_svg(rul, critical, watch):
#     """Half-circle dial. The zone the needle is in stays bright, the others dim."""
#     cx, cy, r = 200, 205, 150
#     value = min(max(rul, 0), RUL_CAP)
#     parts = []
#     zones = [(0, critical, ZONE["CRITICAL"]), (critical, watch, ZONE["WATCH"]),
#              (watch, RUL_CAP, ZONE["HEALTHY"])]
#     for lo, hi, color in zones:
#         if hi <= lo:
#             continue
#         x1, y1 = _polar(cx, cy, r, lo)
#         x2, y2 = _polar(cx, cy, r, hi)
#         active = lo <= value < hi or (hi == RUL_CAP and value >= lo)
#         parts.append(f'<path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 0 1 {x2:.1f} {y2:.1f}" fill="none" '
#                      f'stroke="{color}" stroke-width="22" opacity="{1 if active else 0.3}"/>')
#     for v in range(0, RUL_CAP + 1, 25):
#         xa, ya = _polar(cx, cy, r + 16, v)
#         xb, yb = _polar(cx, cy, r + 23, v)
#         xt, yt = _polar(cx, cy, r + 40, v)
#         parts.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
#                      f'stroke="#6F8190" stroke-width="2"/>')
#         parts.append(f'<text x="{xt:.1f}" y="{yt + 5:.1f}" text-anchor="middle" font-size="14" '
#                      f'font-family="Arial, sans-serif" fill="#9FB0BC">{v}</text>')
#     tip_x, tip_y = _polar(cx, cy, r - 30, value)
#     ang = math.pi * (1 - value / RUL_CAP)
#     dx, dy = 7 * math.sin(ang), 7 * math.cos(ang)
#     parts.append(f'<polygon points="{tip_x:.1f},{tip_y:.1f} {cx + dx:.1f},{cy + dy:.1f} '
#                  f'{cx - dx:.1f},{cy - dy:.1f}" fill="#F2F5F7"/>')
#     parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#F2F5F7"/>'
#                  f'<circle cx="{cx}" cy="{cy}" r="4" fill="#1C2E3B"/>')
#     return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 230">' + "".join(parts) + "</svg>")


# def fan_svg():
#     """Line drawing of a turbofan seen from the front, used once in the page heading."""
#     cx = cy = 100
#     blades = []
#     for i in range(18):
#         a = 2 * math.pi * i / 18
#         x1, y1 = cx + 22 * math.cos(a), cy + 22 * math.sin(a)
#         xc, yc = cx + 58 * math.cos(a + 0.45), cy + 58 * math.sin(a + 0.45)
#         x2, y2 = cx + 87 * math.cos(a + 0.9), cy + 87 * math.sin(a + 0.9)
#         blades.append(f'<path d="M{x1:.1f} {y1:.1f} Q{xc:.1f} {yc:.1f} {x2:.1f} {y2:.1f}"/>')
#     return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" fill="none" '
#             'stroke="#7C8E9B" stroke-width="1.4" stroke-linecap="round">'
#             '<circle cx="100" cy="100" r="96"/><circle cx="100" cy="100" r="90" stroke-width="0.7"/>'
#             '<circle cx="100" cy="100" r="22"/><circle cx="100" cy="100" r="8"/>'
#             + "".join(blades) + "</svg>")


# # Data and model plumbing
# class ModelBundle:
#     """A model together with the preprocessor that knows how to feed it.

#     Everything the UI does goes through the two abstract interfaces, so this
#     class never has to ask whether it is holding XGBoost or an LSTM."""

#     def __init__(self, model, preprocessor, fleet_inputs):
#         self.model = model
#         self.preprocessor = preprocessor
#         self.label = model.name
#         self.fleet_pred = np.clip(model.predict(fleet_inputs), 0, RUL_CAP)

#     def history(self, engine_df):
#         """Predicted RUL at every cycle where the model has enough context."""
#         inputs, cycles = self.preprocessor.engine_inputs(engine_df)
#         preds = np.clip(self.model.predict(inputs), 0, RUL_CAP)
#         return pd.DataFrame({"cycle": cycles, "RUL": preds})


# class Assets:
#     """Loads the dataset once and prepares every model that is available."""

#     def __init__(self):
#         loader = CMAPSSDataLoader(subset="FD001")
#         self.train_df, self.test_df, test_last = loader.load()
#         self.test_last = test_last.sort_values("engine_id").reset_index(drop=True)

#         self.engine_ids = self.test_last["engine_id"].tolist()
#         self.final_rul = dict(zip(self.test_last["engine_id"], self.test_last["RUL"]))
#         self.true_final = np.minimum(self.test_last["RUL"].to_numpy(), RUL_CAP)

#         self.sensor_mean = self.train_df[SENSORS].mean()
#         self.sensor_std = self.train_df[SENSORS].std()

#         self.notes = []
#         self.bundles = []
#         self._add_xgboost()
#         self._add_lstm()

#     def _add_xgboost(self):
#         prep = TabularPreprocessor(rul_cap=RUL_CAP)
#         X_train, y_train, X_test, _ = prep.prepare(self.train_df, self.test_df, self.test_last)
#         model = XGBRULModel(rul_cap=RUL_CAP)
#         if XGB_PATH.exists():
#             model.load(str(XGB_PATH))
#         else:
#             model.fit(X_train, y_train)
#             model.save(str(XGB_PATH))
#             self.notes.append("xgb_model.json was missing, so I trained XGBoost just now and saved it.")
#         self.bundles.append(ModelBundle(model, prep, X_test))

#     def _add_lstm(self):
#         if not LSTM_PATH.exists():
#             self.notes.append("lstm_model.keras not found. Run lstm_model.ipynb to create it "
#                               "and the LSTM will show up here.")
#             return
#         try:
#             prep = SequencePreprocessor(rul_cap=RUL_CAP)
#             _, _, X_test, _ = prep.prepare(self.train_df, self.test_df, self.test_last)
#             model = LSTMRULModel(rul_cap=RUL_CAP).load(str(LSTM_PATH))
#             self.bundles.append(ModelBundle(model, prep, X_test))
#         except Exception as err:
#             self.notes.append(f"The LSTM could not be loaded: {err}")

#     # small helpers used by the tabs
#     def bundle(self, label):
#         return next(b for b in self.bundles if b.label == label)

#     def engine_frame(self, engine_id):
#         df = self.test_df[self.test_df["engine_id"] == engine_id].sort_values("cycle")
#         return df.reset_index(drop=True)

#     def true_rul(self, engine_id, cycles):
#         """Actual RUL at each cycle. The test files only reveal it at the last
#         cycle, but it drops by one every cycle, so we can walk backwards."""
#         cycles = np.asarray(cycles)
#         last_cycle = self.test_df.loc[self.test_df["engine_id"] == engine_id, "cycle"].max()
#         return np.minimum(self.final_rul[engine_id] + (last_cycle - cycles), RUL_CAP)

#     def inject(self, engine_df, sensor, sigma, n_last):
#         """Return a copy where one sensor drifts away from normal over the last n cycles."""
#         out = engine_df.copy()
#         n_last = min(n_last, len(out))
#         tail = out.index[-n_last:]
#         ramp = np.linspace(1 / n_last, 1.0, n_last)
#         out.loc[tail, sensor] = out.loc[tail, sensor] + ramp * sigma * self.sensor_std[sensor]
#         return out


# @st.cache_resource(show_spinner="Loading data and models. The first run takes a little while.")
# def get_assets():
#     return Assets()


# @st.cache_data(show_spinner=False)
# def cached_history(_assets, label, engine_id):
#     # _assets starts with an underscore so streamlit skips hashing it
#     return _assets.bundle(label).history(_assets.engine_frame(engine_id))


# @dataclass
# class Settings:
#     bundles: list
#     engine: int
#     critical: int
#     watch: int


# # The app
# class ConsoleApp:
#     def __init__(self, assets):
#         self.assets = assets
#         self.labels = [b.label for b in assets.bundles]

#     def _pick_random_engine(self):
#         st.session_state["engine"] = random.choice(self.assets.engine_ids)

#     def _sidebar(self):
#         st.sidebar.header("Dashboard controls")
#         chosen = st.sidebar.multiselect("Models", self.labels, default=self.labels)

#         if "engine" not in st.session_state:
#             st.session_state["engine"] = self.assets.engine_ids[0]
#         st.sidebar.selectbox("Engine", self.assets.engine_ids, key="engine")
#         st.sidebar.button("Pick a random engine", on_click=self._pick_random_engine)

#         st.sidebar.subheader("Alert levels")
#         critical = st.sidebar.slider("Critical when RUL is below", 5, 60, 30)
#         watch = st.sidebar.slider("Watch when RUL is below", 20, 110, 70)
#         watch = max(watch, critical + 1)

#         for note in self.assets.notes:
#             st.sidebar.info(note)

#         bundles = [b for b in self.assets.bundles if b.label in chosen]
#         return Settings(bundles, st.session_state["engine"], critical, watch)

#     def run(self):
#         st.subheader("Analyst dashboard")
#         st.caption("Replay a real test engine cycle by cycle, inject faults, compare models across "
#                    "the fleet, or look at the class design. Controls are in the sidebar.")
#         s = self._sidebar()
#         if not s.bundles:
#             st.warning("Choose at least one model in the sidebar.")
#             return

#         replay, fault, fleet, hood = st.tabs(
#             ["Live replay", "Fault injection", "Fleet view", "Under the hood"])
#         with replay:
#             self._tab_replay(s)
#         with fault:
#             self._tab_fault(s)
#         with fleet:
#             self._tab_fleet(s)
#         with hood:
#             self._tab_hood(s)

#     # Tab 1: replay
#     def _tab_replay(self, s):
#         eng = self.assets.engine_frame(s.engine)
#         hist = {b.label: cached_history(self.assets, b.label, s.engine) for b in s.bundles}
#         first_ok = int(max(h["cycle"].min() for h in hist.values()))
#         last = int(eng["cycle"].max())
#         if first_ok >= last:
#             st.info("This engine has too little history for the selected models.")
#             return

#         watch_sensors = st.multiselect(
#             "Sensors to watch", SENSORS, default=["sensor_11", "sensor_4", "sensor_12"])
#         left, right = st.columns([5, 1])
#         with left:
#             cycle = st.slider("Cycle", first_ok, last, last, key=f"cycle_{s.engine}")
#         with right:
#             st.write("")
#             play = st.button("Play replay")

#         slot = st.empty()
#         if play:
#             step = max(1, (last - first_ok) // 70)
#             for c in list(range(first_ok, last + 1, step)) + [last]:
#                 with slot.container():
#                     self._draw_replay(s, eng, hist, watch_sensors, c, first_ok, last)
#                 time.sleep(0.05)
#         else:
#             with slot.container():
#                 self._draw_replay(s, eng, hist, watch_sensors, cycle, first_ok, last)

#     def _draw_replay(self, s, eng, hist, watch_sensors, c, first_ok, last):
#         a = self.assets
#         actual_now = float(a.true_rul(s.engine, [c])[0])

#         cols = st.columns(len(s.bundles) + 1)
#         cols[0].metric("Actual RUL", f"{actual_now:.0f} cycles")
#         cols[0].caption(f"Engine {s.engine}, cycle {c} of {last}")
#         for col, b in zip(cols[1:], s.bundles):
#             h = hist[b.label]
#             pred = float(h.loc[h["cycle"] == c, "RUL"].iloc[0])
#             col.metric(f"{b.label} estimate", f"{pred:.0f} cycles",
#                        delta=f"{pred - actual_now:+.0f} vs actual", delta_color="off")
#             col.markdown(badge(pred, s.critical, s.watch), unsafe_allow_html=True)
#             col.progress(float(min(pred / RUL_CAP, 1.0)))

#         # remaining life: the actual curve is drawn in full, predictions only up to now
#         all_cycles = eng["cycle"].to_numpy()
#         frames = [pd.DataFrame({"cycle": all_cycles, "RUL": a.true_rul(s.engine, all_cycles),
#                                 "series": "Actual"})]
#         for b in s.bundles:
#             h = hist[b.label]
#             frames.append(h[h["cycle"] <= c].assign(series=b.label))
#         data = pd.concat(frames)

#         names = ["Actual"] + [b.label for b in s.bundles]
#         color = alt.Color("series:N", legend=alt.Legend(title=None, orient="top"),
#                           scale=alt.Scale(domain=names, range=[COLORS[n] for n in names]))
#         x_scale = alt.Scale(domain=[first_ok, last])

#         lines = alt.Chart(data).mark_line(strokeWidth=2.5).encode(
#             x=alt.X("cycle:Q", title="Cycle", scale=x_scale),
#             y=alt.Y("RUL:Q", title="Remaining useful life (cycles)",
#                     scale=alt.Scale(domain=[0, RUL_CAP + 5])),
#             color=color)
#         critical_band = alt.Chart(pd.DataFrame({"lo": [0], "hi": [s.critical]})).mark_rect(
#             opacity=0.12, color="#d64545").encode(y="lo:Q", y2="hi:Q")
#         watch_band = alt.Chart(pd.DataFrame({"lo": [s.critical], "hi": [s.watch]})).mark_rect(
#             opacity=0.10, color="#e0a526").encode(y="lo:Q", y2="hi:Q")
#         now_line = alt.Chart(pd.DataFrame({"cycle": [c]})).mark_rule(
#             color="#888888", strokeDash=[4, 4]).encode(x="cycle:Q")
#         st.altair_chart((critical_band + watch_band + lines + now_line)
#                         .properties(height=320, width="container"))

#         if watch_sensors:
#             window = eng.loc[eng["cycle"] <= c, ["cycle"] + watch_sensors].set_index("cycle")
#             z = window.sub(a.sensor_mean[watch_sensors], axis="columns")
#             z = z.div(a.sensor_std[watch_sensors], axis="columns")
#             z = z.reset_index().melt("cycle", var_name="sensor", value_name="z")
#             sensor_chart = alt.Chart(z).mark_line().encode(
#                 x=alt.X("cycle:Q", title="Cycle", scale=x_scale),
#                 y=alt.Y("z:Q", title="Std devs from fleet average"),
#                 color=alt.Color("sensor:N", legend=alt.Legend(title=None, orient="top")))
#             st.altair_chart(sensor_chart.properties(height=200, width="container"))

#     # Tab 2: fault injection
#     def _tab_fault(self, s):
#         st.write("Pick a sensor and make it drift away from normal over the last few cycles of "
#                  "this engine. The models re read the doctored history and tell you how much "
#                  "life they think just disappeared.")
#         c1, c2, c3 = st.columns(3)
#         sensor = c1.selectbox("Sensor", SENSORS, index=SENSORS.index("sensor_11"))
#         sigma = c2.slider("Drift size (std devs)", -4.0, 4.0, 2.0, 0.5)
#         n_last = c3.slider("Applied over the last N cycles", 3, 60, 15)

#         eng = self.assets.engine_frame(s.engine)
#         doctored = self.assets.inject(eng, sensor, sigma, n_last)

#         cols = st.columns(len(s.bundles))
#         for col, b in zip(cols, s.bundles):
#             before = float(cached_history(self.assets, b.label, s.engine)["RUL"].iloc[-1])
#             after = float(b.history(doctored)["RUL"].iloc[-1])
#             col.metric(f"{b.label}: RUL now", f"{after:.0f} cycles",
#                        delta=f"{after - before:+.1f} from {before:.0f}", delta_color="off")
#             col.markdown(badge(after, s.critical, s.watch), unsafe_allow_html=True)

#         trace = pd.concat([
#             eng[["cycle", sensor]].rename(columns={sensor: "value"}).assign(version="Original"),
#             doctored[["cycle", sensor]].rename(columns={sensor: "value"}).assign(version="Doctored"),
#         ])
#         trace = trace[trace["cycle"] >= trace["cycle"].max() - 80]
#         chart = alt.Chart(trace).mark_line().encode(
#             x=alt.X("cycle:Q", title="Cycle"),
#             y=alt.Y("value:Q", title=sensor, scale=alt.Scale(zero=False)),
#             color=alt.Color("version:N", legend=alt.Legend(title=None, orient="top"),
#                             scale=alt.Scale(domain=["Original", "Doctored"],
#                                             range=["#8a8f98", "#d64545"])))
#         st.altair_chart(chart.properties(height=240, width="container"))

#         with st.expander("Which sensor does this engine react to most?"):
#             st.write("Runs the same drift on every informative sensor, one at a time, "
#                      "and records how far the estimate moves.")
#             pick = st.selectbox("Model", [b.label for b in s.bundles], key="sweep_model")
#             if st.button("Run sweep"):
#                 bundle = self.assets.bundle(pick)
#                 base = float(cached_history(self.assets, pick, s.engine)["RUL"].iloc[-1])
#                 rows, bar = [], st.progress(0.0)
#                 for i, name in enumerate(SENSORS):
#                     new = float(bundle.history(
#                         self.assets.inject(eng, name, sigma, n_last))["RUL"].iloc[-1])
#                     rows.append({"sensor": name, "change": new - base, "size": abs(new - base)})
#                     bar.progress((i + 1) / len(SENSORS))
#                 bar.empty()
#                 res = pd.DataFrame(rows)
#                 sweep = alt.Chart(res).mark_bar().encode(
#                     y=alt.Y("sensor:N", sort=alt.EncodingSortField(field="size", order="descending"),
#                             title=None),
#                     x=alt.X("change:Q", title="Change in predicted RUL (cycles)"),
#                     color=alt.condition(alt.datum.change > 0, alt.value("#2e9e5b"),
#                                         alt.value("#d64545")),
#                     tooltip=["sensor", "change"])
#                 st.altair_chart(sweep.properties(height=380, width="container"))

#     # Tab 3: fleet
#     def _tab_fleet(self, s):
#         a = self.assets
#         truth = a.true_final

#         score_rows = []
#         for b in s.bundles:
#             m = RegressionEvaluator.metrics(truth, b.fleet_pred)
#             m["NASA score"] = RegressionEvaluator.nasa_score(truth, b.fleet_pred)
#             score_rows.append(pd.Series(m, name=b.label))
#         st.subheader("How well do the models do on all 100 test engines?")
#         st.dataframe(pd.DataFrame(score_rows).round(3))
#         st.caption("Scored against actual RUL capped at 125 for every model, so the comparison is "
#                    "fair. Lower is better for everything except R2. The notebook scores the LSTM "
#                    "against the uncapped value, so its numbers will differ a little.")

#         pts = pd.concat([
#             pd.DataFrame({"engine": a.engine_ids, "Actual": truth,
#                           "Predicted": b.fleet_pred, "model": b.label})
#             for b in s.bundles])
#         names = [b.label for b in s.bundles]
#         dots = alt.Chart(pts).mark_circle(size=70, opacity=0.75).encode(
#             x=alt.X("Actual:Q", title="Actual RUL"),
#             y=alt.Y("Predicted:Q", title="Predicted RUL"),
#             color=alt.Color("model:N", legend=alt.Legend(title=None, orient="top"),
#                             scale=alt.Scale(domain=names, range=[COLORS[n] for n in names])),
#             tooltip=["engine", "model", "Actual", "Predicted"])
#         diagonal = alt.Chart(pd.DataFrame({"v": [0, RUL_CAP]})).mark_line(
#             color="#888888", strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
#         st.altair_chart((diagonal + dots).properties(height=380, width="container"))
#         st.caption("Dots on the dashed line are perfect predictions.")

#         st.subheader("Maintenance queue")
#         queue = pd.DataFrame({"Engine": a.engine_ids, "Actual RUL": truth})
#         for b in s.bundles:
#             queue[b.label] = b.fleet_pred
#         worst = queue[[b.label for b in s.bundles]].min(axis=1)
#         queue["Status"] = [status_for(v, s.critical, s.watch)[0] for v in worst]
#         queue = queue.assign(_sort=worst).sort_values("_sort").drop(columns="_sort").head(15)
#         config = {b.label: st.column_config.ProgressColumn(
#             b.label, min_value=0, max_value=RUL_CAP, format="%d") for b in s.bundles}
#         st.dataframe(queue, hide_index=True, column_config=config)
#         st.caption("The 15 engines closest to failure according to the most pessimistic selected model.")

#         counts = pd.DataFrame({
#             b.label: {"Critical": int((b.fleet_pred <= s.critical).sum()),
#                       "Watch": int(((b.fleet_pred > s.critical) & (b.fleet_pred <= s.watch)).sum()),
#                       "Healthy": int((b.fleet_pred > s.watch).sum())}
#             for b in s.bundles})
#         st.write("Fleet status by model")
#         st.dataframe(counts.T)

#         if len(s.bundles) >= 2:
#             first, second = s.bundles[0], s.bundles[1]
#             gap = pd.DataFrame({"Engine": a.engine_ids, "Actual RUL": truth,
#                                 first.label: first.fleet_pred, second.label: second.fleet_pred})
#             gap["Gap"] = (gap[first.label] - gap[second.label]).abs()
#             st.write(f"Engines where {first.label} and {second.label} disagree the most")
#             st.dataframe(gap.sort_values("Gap", ascending=False).head(5).round(1), hide_index=True)

#     # Tab 4: OOP showcase
#     def _tab_hood(self, s):
#         st.write("Everything above runs through two abstract base classes. The UI only ever calls "
#                  "`engine_inputs`, `fit`, `predict` and `load`, and each concrete class decides "
#                  "what that means for itself.")

#         st.subheader("Inheritance and abstraction")
#         for b in self.assets.bundles:
#             model_chain = " > ".join(c.__name__ for c in type(b.model).__mro__)
#             prep_chain = " > ".join(c.__name__ for c in type(b.preprocessor).__mro__)
#             st.code(f"{model_chain}\n{prep_chain}", language="text")

#         st.subheader("Polymorphism: who overrides what")
#         model_steps = ["_build_model", "_fit_model", "_predict_raw", "_postprocess",
#                        "save", "load", "fit", "predict", "evaluate"]
#         st.dataframe(self._override_table(BaseRULModel, [type(b.model) for b in self.assets.bundles],
#                                           model_steps), hide_index=True)
#         prep_steps = ["prepare", "engine_inputs", "_clean", "find_constant_columns"]
#         st.dataframe(self._override_table(BaseRULPreprocessor,
#                                           [type(b.preprocessor) for b in self.assets.bundles],
#                                           prep_steps), hide_index=True)

#         st.subheader("Encapsulation: try to break a model")
#         st.write("`rul_cap` is private behind a property that validates the value. "
#                  "Try something silly on a throwaway model.")
#         value = st.number_input("Set rul_cap to", value=-5, step=5)
#         scratch = XGBRULModel()
#         try:
#             scratch.rul_cap = value
#             st.success(f"Accepted. rul_cap is now {scratch.rul_cap}.")
#         except ValueError as err:
#             st.error(f"Rejected by the setter: {err}")
#         try:
#             BaseRULModel("direct")
#         except TypeError as err:
#             st.caption(f"Abstraction check, building the base class directly fails: {err}")

#     @staticmethod
#     def _override_table(base, classes, steps):
#         rows = []
#         for step in steps:
#             row = {"method": step,
#                    base.__name__: "abstract" if step in base.__abstractmethods__ else "implemented"}
#             for cls in classes:
#                 row[cls.__name__] = "overridden" if step in vars(cls) else "inherited"
#             rows.append(row)
#         return pd.DataFrame(rows)


# # Predict page: the main thing most people will use
# class InputParser:
#     """Turns whatever the user gives us into the frame the preprocessors expect.

#     Only the 17 informative columns are required. engine_id and cycle are optional,
#     and the constant columns the models never use are filled in automatically."""

#     REQUIRED = ["setting_1", "setting_2"] + SENSORS

#     @staticmethod
#     def _numeric_labels(columns):
#         # a real header has letters in it; a first row of numbers does not
#         return not any(any(ch.isalpha() for ch in str(c)) for c in columns)

#     @classmethod
#     def read_csv(cls, file):
#         """Read a CSV (comma, semicolon or tab) or a raw NASA .txt file with no header row."""
#         raw = file.getvalue()
#         df, used = None, None
#         for kw in ({"sep": ","}, {"sep": ";"}, {"sep": "\t"}, {"sep": r"\s+"}):
#             try:
#                 cand = pd.read_csv(io.BytesIO(raw), **kw)
#             except Exception:
#                 continue
#             if cand.shape[1] > 1:
#                 df, used = cand, kw
#                 break
#         if df is None:
#             raise ValueError("This file could not be read as a table. Use a CSV with a header row.")
#         if cls._numeric_labels(df.columns):          # first row is data, so there is no header
#             df = pd.read_csv(io.BytesIO(raw), header=None, **used)
#             if df.shape[1] != len(CMAPSSDataLoader.ALL_COLS):
#                 raise ValueError("The file has no header row. Add one using the column names "
#                                  "from the template.")
#             df.columns = CMAPSSDataLoader.ALL_COLS
#         return df

#     @classmethod
#     def clean(cls, df):
#         """Return (frame in the training column order, list of notes for the user)."""
#         notes = []
#         df = df.copy()
#         df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

#         missing = [c for c in cls.REQUIRED if c not in df.columns]
#         if missing:
#             raise ValueError("These columns are missing: " + ", ".join(missing) +
#                              ". Download the template to see the expected layout.")
#         if "engine_id" not in df.columns:
#             df["engine_id"] = 1
#         if "cycle" not in df.columns:
#             df["cycle"] = df.groupby("engine_id").cumcount() + 1
#             notes.append("There is no cycle column, so the rows were treated as cycles "
#                          "1, 2, 3 ... in the order given.")

#         df = df[["engine_id", "cycle"] + cls.REQUIRED].apply(pd.to_numeric, errors="coerce")
#         df = df.dropna(subset=["engine_id", "cycle"]).sort_values(["engine_id", "cycle"])
#         if df.empty:
#             raise ValueError("The table has no usable rows. Every row needs a numeric cycle.")

#         blanks = int(df[cls.REQUIRED].isna().sum().sum())
#         if blanks:
#             df[cls.REQUIRED] = df.groupby("engine_id")[cls.REQUIRED].transform(
#                 lambda s: s.ffill().bfill())
#             if df[cls.REQUIRED].isna().any().any():
#                 raise ValueError("At least one column has no numbers in it at all.")
#             notes.append(f"{blanks} blank or non numeric cells were filled with the nearest "
#                          "reading from the same engine.")

#         df["engine_id"] = df["engine_id"].astype(int)
#         for col in BaseRULPreprocessor.CONSTANT_COLS:     # dropped again by the preprocessors
#             df[col] = 0.0
#         return df[CMAPSSDataLoader.ALL_COLS].reset_index(drop=True), notes


# class PredictPage:
#     MODES = ["Upload a CSV", "Enter readings manually", "Try an example"]
#     ADVICE = {
#         "CRITICAL": "Schedule maintenance as soon as possible.",
#         "WATCH": "Plan maintenance in the coming cycles.",
#         "HEALTHY": "No action needed right now.",
#     }

#     def __init__(self, assets):
#         self.assets = assets
#         self.labels = [b.label for b in assets.bundles]
#         self.sample = assets.engine_frame(assets.engine_ids[0])
#         self._example_id = None

#     # small helpers
#     def _template_frame(self, with_id=True):
#         cols = (["engine_id"] if with_id else []) + ["cycle"] + InputParser.REQUIRED
#         return self.sample.tail(30)[cols].reset_index(drop=True)

#     @staticmethod
#     def _min_cycles(bundle):
#         # the LSTM needs a full window, XGBoost works with any amount of history
#         return getattr(bundle.preprocessor, "seq_length", 1)

#     def _pick_example(self):
#         st.session_state["example_engine"] = random.choice(self.assets.engine_ids)

#     # page
#     def run(self):
#         st.markdown(
#             '<div class="hero"><div><h1>How many flights does this engine have left?</h1>'
#             "<p>Add the sensor readings from each flight. We estimate how many more cycles "
#             "it can fly before maintenance is due.</p></div>"
#             + svg_img(fan_svg(), "fan") + "</div>",
#             unsafe_allow_html=True)

#         st.subheader("Your engine data")
#         mode = st.radio("How do you want to add it?", self.MODES, horizontal=True,
#                         label_visibility="collapsed")
#         raw = {"Upload a CSV": self._from_upload,
#                "Enter readings manually": self._from_table,
#                "Try an example": self._from_example}[mode]()
#         if raw is None:
#             return

#         try:
#             data, notes = InputParser.clean(raw)
#         except ValueError as err:
#             st.error(str(err))
#             return
#         for note in notes:
#             st.info(note)

#         bundle, critical, watch = self._settings()
#         engines = sorted(data["engine_id"].unique())
#         if len(engines) == 1:
#             actual = None
#             if mode == "Try an example":
#                 actual = float(self.assets.true_rul(self._example_id, [data["cycle"].max()])[0])
#             self._show_single(bundle, data, critical, watch, actual)
#         else:
#             self._show_many(bundle, data, critical, watch)
#         self._options_panel()

#     # three ways to give data
#     def _from_upload(self):
#         file = st.file_uploader("Sensor readings", type=["csv", "txt"])
#         st.download_button("Download a template CSV",
#                            self._template_frame().to_csv(index=False).encode(),
#                            "engine_template.csv", "text/csv")
#         with st.expander("What should the file contain?"):
#             st.markdown(
#                 "One row per flight cycle, with these columns:\n\n"
#                 f"- `cycle`: the flight number, counting from the engine's first flight\n"
#                 f"- `setting_1`, `setting_2`\n"
#                 f"- {', '.join(f'`{c}`' for c in SENSORS)}\n"
#                 "- `engine_id` (optional): add it to score several engines in one file\n\n"
#                 "Aim for at least 30 cycles per engine. The models were trained on the NASA "
#                 "C-MAPSS FD001 turbofan data, so they are only reliable for engines that "
#                 "behave like it. Raw NASA `.txt` files with no header also work.")
#         if file is None:
#             st.info("Upload a file to get a prediction, or try an example first.")
#             return None
#         try:
#             return InputParser.read_csv(file)
#         except Exception as err:
#             st.error(f"Could not read the file: {err}")
#             return None

#     def _from_table(self):
#         st.caption("The table starts with example values. Type your own readings over them, "
#                    "one row per flight cycle, and use the + at the bottom to add rows.")
#         return st.data_editor(self._template_frame(with_id=False), num_rows="dynamic",
#                               key="manual_table")

#     def _from_example(self):
#         if "example_engine" not in st.session_state:
#             self._pick_example()
#         self._example_id = st.session_state["example_engine"]
#         st.button("Try a different example", on_click=self._pick_example)
#         st.caption(f"Using engine {self._example_id} from the NASA test set. "
#                    "The models have never seen it.")
#         return self.assets.engine_frame(self._example_id)

#     # model + alert levels, tucked away below the result
#     def _settings(self):
#         """Read the options from session state so they can live below the result."""
#         label = st.session_state.get("pred_model", self.labels[0])
#         critical = st.session_state.get("pred_critical", 30)
#         watch = st.session_state.get("pred_watch", 70)
#         return self.assets.bundle(label), critical, max(watch, critical + 1)

#     def _options_panel(self):
#         with st.expander("Options: model and alert levels"):
#             c1, c2, c3 = st.columns(3)
#             c1.selectbox("Model", self.labels, key="pred_model")
#             c2.slider("Critical below", 5, 60, 30, key="pred_critical")
#             c3.slider("Watch below", 20, 110, 70, key="pred_watch")
#             st.caption("XGBoost works with any amount of history. The LSTM needs at least "
#                        "30 cycles per engine.")

#     # results
#     def _show_single(self, bundle, engine_df, critical, watch, actual=None):
#         need, n = self._min_cycles(bundle), len(engine_df)
#         if n < need:
#             st.warning(f"The {bundle.label} model needs at least {need} cycles and you gave {n}. "
#                        "Add more rows, or pick XGBoost under Options.")
#             return
#         try:
#             hist = bundle.history(engine_df)
#         except Exception as err:
#             st.error(f"Could not make a prediction from this data: {err}")
#             return

#         rul = float(hist["RUL"].iloc[-1])
#         status = status_for(rul, critical, watch)[0]
#         capped = rul >= RUL_CAP - 0.5
#         shown = f"{RUL_CAP}+" if capped else f"{rul:.0f}"

#         facts = [("Model", bundle.label), ("Cycles in your data", str(n)),
#                  ("Latest cycle", str(int(engine_df["cycle"].max()))),
#                  ("Alert levels", f"Critical under {critical}, watch under {watch}")]
#         if actual is not None:
#             facts.append(("Actual remaining life", f"{min(actual, RUL_CAP):.0f} cycles"))
#         facts_html = "".join(f"<div><b>{k}</b>{v}</div>" for k, v in facts)

#         st.markdown(
#             '<div class="panel"><div class="panel-top">'
#             + svg_img(gauge_svg(rul, critical, watch), "gauge",
#                       f"Gauge showing about {shown} cycles of life left")
#             + '<div class="readout"><div class="small">Estimated remaining useful life</div>'
#             f'<div class="num">{shown}<span>flight cycles</span></div>'
#             f'<span class="pill" style="background:{ZONE[status]}">{status.capitalize()}</span>'
#             f'<div class="advice">{self.ADVICE[status]}</div></div></div>'
#             f'<div class="facts">{facts_html}</div></div>',
#             unsafe_allow_html=True)

#         if capped:
#             st.caption(f"Estimates stop at {RUL_CAP} cycles. The engine is healthy and the model "
#                        "can not see further ahead than that.")
#         if n < 30:
#             st.caption(f"Only {n} cycles were given. Estimates are more reliable with 30 or more.")

#         with st.expander("See how the estimate changed over time"):
#             lines = alt.Chart(hist).mark_line(strokeWidth=2.5, color=COLORS[bundle.label]).encode(
#                 x=alt.X("cycle:Q", title="Cycle", scale=alt.Scale(zero=False)),
#                 y=alt.Y("RUL:Q", title="Estimated remaining life (cycles)",
#                         scale=alt.Scale(domain=[0, RUL_CAP + 5])))
#             crit = alt.Chart(pd.DataFrame({"lo": [0], "hi": [critical]})).mark_rect(
#                 opacity=0.14, color=ZONE["CRITICAL"]).encode(y="lo:Q", y2="hi:Q")
#             wat = alt.Chart(pd.DataFrame({"lo": [critical], "hi": [watch]})).mark_rect(
#                 opacity=0.14, color=ZONE["WATCH"]).encode(y="lo:Q", y2="hi:Q")
#             st.altair_chart((crit + wat + lines).properties(height=280, width="container"))
#             st.caption("Red and yellow bands show the alert levels from Options.")

#     def _show_many(self, bundle, data, critical, watch):
#         need = self._min_cycles(bundle)
#         rows = []
#         for engine_id, g in data.groupby("engine_id"):
#             rul = np.nan
#             if len(g) >= need:
#                 try:
#                     rul = float(bundle.history(g.reset_index(drop=True))["RUL"].iloc[-1])
#                 except Exception:
#                     pass
#             rows.append({"Engine": int(engine_id), "Cycles given": len(g), "Estimated RUL": rul,
#                          "Status": "Not enough data" if np.isnan(rul)
#                          else status_for(rul, critical, watch)[0].capitalize()})
#         table = pd.DataFrame(rows).sort_values("Estimated RUL", na_position="last")

#         counts = table["Status"].value_counts()
#         st.subheader(f"{len(table)} engines, closest to failure first")
#         st.markdown(f"**{counts.get('Critical', 0)}** critical, **{counts.get('Watch', 0)}** on "
#                     f"watch, **{counts.get('Healthy', 0)}** healthy.")
#         st.dataframe(table, hide_index=True, column_config={
#             "Estimated RUL": st.column_config.ProgressColumn(
#                 "Estimated RUL (cycles)", min_value=0, max_value=RUL_CAP, format="%d")})
#         st.download_button("Download results as CSV", table.round(1).to_csv(index=False).encode(),
#                            "rul_predictions.csv", "text/csv")

#         usable = table.dropna(subset=["Estimated RUL"])["Engine"].tolist()
#         if usable:
#             pick = st.selectbox("Look closer at one engine", usable)
#             self._show_single(bundle, data[data["engine_id"] == pick].reset_index(drop=True),
#                               critical, watch)


# def _navigation():
#     options = ["Predict", "Analyst dashboard"]
#     if hasattr(st, "segmented_control"):
#         view = st.segmented_control("View", options, default=options[0],
#                                     label_visibility="collapsed")
#     else:
#         view = st.radio("View", options, horizontal=True, label_visibility="collapsed")
#     return view or options[0]


# def main():
#     st.set_page_config(page_title="Engine Life Predictor", layout="wide")
#     st.markdown(PAGE_STYLE, unsafe_allow_html=True)
#     assets = get_assets()

#     if _navigation() == "Predict":
#         _, middle, _ = st.columns([1, 6, 1])      # keeps the simple page narrow and readable
#         with middle:
#             PredictPage(assets).run()
#     else:
#         ConsoleApp(assets).run()


# if __name__ == "__main__":
#     main()



"""Engine Life Predictor

A Streamlit front end for the C-MAPSS FD001 remaining useful life models.

  * Predict            - the main page. Upload a CSV or type in sensor readings and get
                         one clear answer: how many flight cycles the engine has left.
  * Analyst dashboard  - the replay / fault injection / fleet / OOP view, kept as a
                         second option for analysts and developers.

Run it from the project root (the folder that has src/ and CMaps/):

    streamlit run app.py

It expects lstm_model.keras and xgb_model.json next to this file. Both are
written by the notebooks. If xgb_model.json is missing the app trains and
saves one itself, if lstm_model.keras is missing the app runs XGBoost only.
"""
import base64
import io
import math
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

ZONE = {"CRITICAL": "#E5594A", "WATCH": "#E9B23A", "HEALTHY": "#4FB27E"}

PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600&display=swap');

/* type: Barlow for reading, Barlow Condensed for headings and big numbers.
   Only text elements are targeted so Streamlit's icon fonts keep working. */
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea {
    font-family: 'Barlow', 'Segoe UI', Arial, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, [data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif;
    font-weight: 600;
}
.stApp h3 { font-size: 1.7rem; letter-spacing: 0.01em; }
.block-container { padding-top: 4.5rem; }
.stButton > button, .stDownloadButton > button { border-radius: 4px; }

/* page heading */
.hero {
    display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
    padding: 0.2rem 0 1.2rem 0; margin-bottom: 1.3rem;
    border-bottom: 3px solid #2E94B3;
}
.stApp .hero h1 { font-size: 3.2rem; line-height: 1.05; margin: 0; padding: 0; max-width: 24ch; }
.hero p { margin: 0.7rem 0 0 0; max-width: 54ch; opacity: 0.78; font-size: 1.05rem; }
.hero .fan { width: 128px; height: 128px; flex: none; }
@media (max-width: 640px) { .hero .fan { display: none; } .stApp .hero h1 { font-size: 2.4rem; } }

/* the result, laid out like an instrument panel */
.panel {
    background: #213544; color: #EEF2F4; border-radius: 6px;
    border: 1px solid rgba(159,176,188,0.22);
    padding: 1.6rem 1.9rem 0 1.9rem; margin: 0.4rem 0 1rem 0;
}
.panel-top { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem 2.6rem; }
.panel .gauge { width: 380px; max-width: 100%; height: auto; flex: 0 1 380px; }
.panel .readout { flex: 1 1 240px; }
.panel .small { color: #9FB0BC; font-size: 1rem; }
.panel .num {
    font-family: 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif;
    font-size: 5.6rem; font-weight: 600; line-height: 1.02;
}
.panel .num span { font-size: 1.4rem; font-weight: 500; color: #9FB0BC; margin-left: 0.5rem; }
.panel .pill {
    display: inline-block; padding: 2px 12px; border-radius: 4px;
    color: #10202B; font-weight: 600; font-size: 0.95rem;
}
.panel .advice { font-size: 1.15rem; margin: 0.8rem 0 0.3rem 0; }
.panel .facts { display: flex; flex-wrap: wrap; margin-top: 1.4rem; border-top: 1px solid rgba(255,255,255,0.15); }
.panel .facts div { flex: 1 1 150px; padding: 0.85rem 0 1.1rem 0; font-size: 1.05rem; }
.panel .facts div + div { padding-left: 1.2rem; border-left: 1px solid rgba(255,255,255,0.15); }
.panel .facts b { display: block; font-weight: 500; color: #9FB0BC; font-size: 0.85rem; }

/* small status label used by the analyst dashboard */
.badge {
    display: inline-block; padding: 2px 11px; border-radius: 4px;
    color: white; font-size: 0.85rem; font-weight: 600;
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
    return f'<span class="badge" style="background:{color}">{text.capitalize()}</span>'


def svg_img(svg, css_class="", alt=""):
    """Embed an SVG as an <img>, which every Streamlit version renders reliably."""
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f'<img class="{css_class}" src="{uri}" alt="{alt}">'


def _polar(cx, cy, radius, value):
    """Point on the dial: RUL 0 sits on the far left, RUL_CAP on the far right."""
    angle = math.pi * (1 - min(max(value, 0), RUL_CAP) / RUL_CAP)
    return cx + radius * math.cos(angle), cy - radius * math.sin(angle)


def gauge_svg(rul, critical, watch):
    """Half-circle dial. The zone the needle is in stays bright, the others dim."""
    cx, cy, r = 200, 205, 150
    value = min(max(rul, 0), RUL_CAP)
    parts = []
    zones = [(0, critical, ZONE["CRITICAL"]), (critical, watch, ZONE["WATCH"]),
             (watch, RUL_CAP, ZONE["HEALTHY"])]
    for lo, hi, color in zones:
        if hi <= lo:
            continue
        x1, y1 = _polar(cx, cy, r, lo)
        x2, y2 = _polar(cx, cy, r, hi)
        active = lo <= value < hi or (hi == RUL_CAP and value >= lo)
        parts.append(f'<path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 0 1 {x2:.1f} {y2:.1f}" fill="none" '
                     f'stroke="{color}" stroke-width="22" opacity="{1 if active else 0.3}"/>')
    for v in range(0, RUL_CAP + 1, 25):
        xa, ya = _polar(cx, cy, r + 16, v)
        xb, yb = _polar(cx, cy, r + 23, v)
        xt, yt = _polar(cx, cy, r + 40, v)
        parts.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
                     f'stroke="#6F8190" stroke-width="2"/>')
        parts.append(f'<text x="{xt:.1f}" y="{yt + 5:.1f}" text-anchor="middle" font-size="14" '
                     f'font-family="Arial, sans-serif" fill="#9FB0BC">{v}</text>')
    tip_x, tip_y = _polar(cx, cy, r - 30, value)
    ang = math.pi * (1 - value / RUL_CAP)
    dx, dy = 7 * math.sin(ang), 7 * math.cos(ang)
    parts.append(f'<polygon points="{tip_x:.1f},{tip_y:.1f} {cx + dx:.1f},{cy + dy:.1f} '
                 f'{cx - dx:.1f},{cy - dy:.1f}" fill="#F2F5F7"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#F2F5F7"/>'
                 f'<circle cx="{cx}" cy="{cy}" r="4" fill="#213544"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 230">' + "".join(parts) + "</svg>")


def fan_svg():
    """Line drawing of a turbofan seen from the front, used once in the page heading."""
    cx = cy = 100
    blades = []
    for i in range(18):
        a = 2 * math.pi * i / 18
        x1, y1 = cx + 22 * math.cos(a), cy + 22 * math.sin(a)
        xc, yc = cx + 58 * math.cos(a + 0.45), cy + 58 * math.sin(a + 0.45)
        x2, y2 = cx + 87 * math.cos(a + 0.9), cy + 87 * math.sin(a + 0.9)
        blades.append(f'<path d="M{x1:.1f} {y1:.1f} Q{xc:.1f} {yc:.1f} {x2:.1f} {y2:.1f}"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" fill="none" '
            'stroke="#7C8E9B" stroke-width="1.4" stroke-linecap="round">'
            '<circle cx="100" cy="100" r="96"/><circle cx="100" cy="100" r="90" stroke-width="0.7"/>'
            '<circle cx="100" cy="100" r="22"/><circle cx="100" cy="100" r="8"/>'
            + "".join(blades) + "</svg>")


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
        st.sidebar.header("Dashboard controls")
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
        st.subheader("Analyst dashboard")
        st.caption("Replay a real test engine cycle by cycle, inject faults, compare models across "
                   "the fleet, or look at the class design. Controls are in the sidebar.")
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


# Predict page: the main thing most people will use
class InputParser:
    """Turns whatever the user gives us into the frame the preprocessors expect.

    Only the 17 informative columns are required. engine_id and cycle are optional,
    and the constant columns the models never use are filled in automatically."""

    REQUIRED = ["setting_1", "setting_2"] + SENSORS

    @staticmethod
    def _numeric_labels(columns):
        # a real header has letters in it; a first row of numbers does not
        return not any(any(ch.isalpha() for ch in str(c)) for c in columns)

    @classmethod
    def read_csv(cls, file):
        """Read a CSV (comma, semicolon or tab) or a raw NASA .txt file with no header row."""
        raw = file.getvalue()
        df, used = None, None
        for kw in ({"sep": ","}, {"sep": ";"}, {"sep": "\t"}, {"sep": r"\s+"}):
            try:
                cand = pd.read_csv(io.BytesIO(raw), **kw)
            except Exception:
                continue
            if cand.shape[1] > 1:
                df, used = cand, kw
                break
        if df is None:
            raise ValueError("This file could not be read as a table. Use a CSV with a header row.")
        if cls._numeric_labels(df.columns):          # first row is data, so there is no header
            df = pd.read_csv(io.BytesIO(raw), header=None, **used)
            if df.shape[1] != len(CMAPSSDataLoader.ALL_COLS):
                raise ValueError("The file has no header row. Add one using the column names "
                                 "from the template.")
            df.columns = CMAPSSDataLoader.ALL_COLS
        return df

    @classmethod
    def clean(cls, df):
        """Return (frame in the training column order, list of notes for the user)."""
        notes = []
        df = df.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        missing = [c for c in cls.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError("These columns are missing: " + ", ".join(missing) +
                             ". Download the template to see the expected layout.")
        if "engine_id" not in df.columns:
            df["engine_id"] = 1
        if "cycle" not in df.columns:
            df["cycle"] = df.groupby("engine_id").cumcount() + 1
            notes.append("There is no cycle column, so the rows were treated as cycles "
                         "1, 2, 3 ... in the order given.")

        df = df[["engine_id", "cycle"] + cls.REQUIRED].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["engine_id", "cycle"]).sort_values(["engine_id", "cycle"])
        if df.empty:
            raise ValueError("The table has no usable rows. Every row needs a numeric cycle.")

        blanks = int(df[cls.REQUIRED].isna().sum().sum())
        if blanks:
            df[cls.REQUIRED] = df.groupby("engine_id")[cls.REQUIRED].transform(
                lambda s: s.ffill().bfill())
            if df[cls.REQUIRED].isna().any().any():
                raise ValueError("At least one column has no numbers in it at all.")
            notes.append(f"{blanks} blank or non numeric cells were filled with the nearest "
                         "reading from the same engine.")

        df["engine_id"] = df["engine_id"].astype(int)
        for col in BaseRULPreprocessor.CONSTANT_COLS:     # dropped again by the preprocessors
            df[col] = 0.0
        return df[CMAPSSDataLoader.ALL_COLS].reset_index(drop=True), notes


class PredictPage:
    MODES = ["Upload a CSV", "Enter readings manually", "Try an example"]
    ADVICE = {
        "CRITICAL": "Schedule maintenance as soon as possible.",
        "WATCH": "Plan maintenance in the coming cycles.",
        "HEALTHY": "No action needed right now.",
    }

    def __init__(self, assets):
        self.assets = assets
        self.labels = [b.label for b in assets.bundles]
        self.sample = assets.engine_frame(assets.engine_ids[0])
        self._example_id = None

    # small helpers
    def _template_frame(self, with_id=True):
        cols = (["engine_id"] if with_id else []) + ["cycle"] + InputParser.REQUIRED
        return self.sample.tail(30)[cols].reset_index(drop=True)

    @staticmethod
    def _min_cycles(bundle):
        # the LSTM needs a full window, XGBoost works with any amount of history
        return getattr(bundle.preprocessor, "seq_length", 1)

    def _pick_example(self):
        st.session_state["example_engine"] = random.choice(self.assets.engine_ids)

    # page
    def run(self):
        st.markdown(
            '<div class="hero"><div><h1>How many flights does this engine have left?</h1>'
            "<p>Add the sensor readings from each flight. We estimate how many more cycles "
            "it can fly before maintenance is due.</p></div>"
            + svg_img(fan_svg(), "fan") + "</div>",
            unsafe_allow_html=True)

        st.subheader("Your engine data")
        mode = st.radio("How do you want to add it?", self.MODES, horizontal=True,
                        label_visibility="collapsed")
        raw = {"Upload a CSV": self._from_upload,
               "Enter readings manually": self._from_table,
               "Try an example": self._from_example}[mode]()
        if raw is None:
            return

        try:
            data, notes = InputParser.clean(raw)
        except ValueError as err:
            st.error(str(err))
            return
        for note in notes:
            st.info(note)

        bundle, critical, watch = self._settings()
        engines = sorted(data["engine_id"].unique())
        if len(engines) == 1:
            actual = None
            if mode == "Try an example":
                actual = float(self.assets.true_rul(self._example_id, [data["cycle"].max()])[0])
            self._show_single(bundle, data, critical, watch, actual)
        else:
            self._show_many(bundle, data, critical, watch)
        self._options_panel()

    # three ways to give data
    def _from_upload(self):
        file = st.file_uploader("Sensor readings", type=["csv", "txt"])
        st.download_button("Download a template CSV",
                           self._template_frame().to_csv(index=False).encode(),
                           "engine_template.csv", "text/csv")
        with st.expander("What should the file contain?"):
            st.markdown(
                "One row per flight cycle, with these columns:\n\n"
                f"- `cycle`: the flight number, counting from the engine's first flight\n"
                f"- `setting_1`, `setting_2`\n"
                f"- {', '.join(f'`{c}`' for c in SENSORS)}\n"
                "- `engine_id` (optional): add it to score several engines in one file\n\n"
                "Aim for at least 30 cycles per engine. The models were trained on the NASA "
                "C-MAPSS FD001 turbofan data, so they are only reliable for engines that "
                "behave like it. Raw NASA `.txt` files with no header also work.")
        if file is None:
            st.info("Upload a file to get a prediction, or try an example first.")
            return None
        try:
            return InputParser.read_csv(file)
        except Exception as err:
            st.error(f"Could not read the file: {err}")
            return None

    def _from_table(self):
        st.caption("The table starts with example values. Type your own readings over them, "
                   "one row per flight cycle, and use the + at the bottom to add rows.")
        return st.data_editor(self._template_frame(with_id=False), num_rows="dynamic",
                              key="manual_table")

    def _from_example(self):
        if "example_engine" not in st.session_state:
            self._pick_example()
        self._example_id = st.session_state["example_engine"]
        st.button("Try a different example", on_click=self._pick_example)
        st.caption(f"Using engine {self._example_id} from the NASA test set. "
                   "The models have never seen it.")
        return self.assets.engine_frame(self._example_id)

    # model + alert levels, tucked away below the result
    def _settings(self):
        """Read the options from session state so they can live below the result."""
        label = st.session_state.get("pred_model", self.labels[0])
        critical = st.session_state.get("pred_critical", 30)
        watch = st.session_state.get("pred_watch", 70)
        return self.assets.bundle(label), critical, max(watch, critical + 1)

    def _options_panel(self):
        with st.expander("Options: model and alert levels"):
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Model", self.labels, key="pred_model")
            c2.slider("Critical below", 5, 60, 30, key="pred_critical")
            c3.slider("Watch below", 20, 110, 70, key="pred_watch")
            st.caption("XGBoost works with any amount of history. The LSTM needs at least "
                       "30 cycles per engine.")

    # results
    def _show_single(self, bundle, engine_df, critical, watch, actual=None):
        need, n = self._min_cycles(bundle), len(engine_df)
        if n < need:
            st.warning(f"The {bundle.label} model needs at least {need} cycles and you gave {n}. "
                       "Add more rows, or pick XGBoost under Options.")
            return
        try:
            hist = bundle.history(engine_df)
        except Exception as err:
            st.error(f"Could not make a prediction from this data: {err}")
            return

        rul = float(hist["RUL"].iloc[-1])
        status = status_for(rul, critical, watch)[0]
        capped = rul >= RUL_CAP - 0.5
        shown = f"{RUL_CAP}+" if capped else f"{rul:.0f}"

        facts = [("Model", bundle.label), ("Cycles in your data", str(n)),
                 ("Latest cycle", str(int(engine_df["cycle"].max()))),
                 ("Alert levels", f"Critical under {critical}, watch under {watch}")]
        if actual is not None:
            facts.append(("Actual remaining life", f"{min(actual, RUL_CAP):.0f} cycles"))
        facts_html = "".join(f"<div><b>{k}</b>{v}</div>" for k, v in facts)

        st.markdown(
            '<div class="panel"><div class="panel-top">'
            + svg_img(gauge_svg(rul, critical, watch), "gauge",
                      f"Gauge showing about {shown} cycles of life left")
            + '<div class="readout"><div class="small">Estimated remaining useful life</div>'
            f'<div class="num">{shown}<span>flight cycles</span></div>'
            f'<span class="pill" style="background:{ZONE[status]}">{status.capitalize()}</span>'
            f'<div class="advice">{self.ADVICE[status]}</div></div></div>'
            f'<div class="facts">{facts_html}</div></div>',
            unsafe_allow_html=True)

        if capped:
            st.caption(f"Estimates stop at {RUL_CAP} cycles. The engine is healthy and the model "
                       "can not see further ahead than that.")
        if n < 30:
            st.caption(f"Only {n} cycles were given. Estimates are more reliable with 30 or more.")

        with st.expander("See how the estimate changed over time"):
            lines = alt.Chart(hist).mark_line(strokeWidth=2.5, color=COLORS[bundle.label]).encode(
                x=alt.X("cycle:Q", title="Cycle", scale=alt.Scale(zero=False)),
                y=alt.Y("RUL:Q", title="Estimated remaining life (cycles)",
                        scale=alt.Scale(domain=[0, RUL_CAP + 5])))
            crit = alt.Chart(pd.DataFrame({"lo": [0], "hi": [critical]})).mark_rect(
                opacity=0.14, color=ZONE["CRITICAL"]).encode(y="lo:Q", y2="hi:Q")
            wat = alt.Chart(pd.DataFrame({"lo": [critical], "hi": [watch]})).mark_rect(
                opacity=0.14, color=ZONE["WATCH"]).encode(y="lo:Q", y2="hi:Q")
            st.altair_chart((crit + wat + lines).properties(height=280, width="container"))
            st.caption("Red and yellow bands show the alert levels from Options.")

    def _show_many(self, bundle, data, critical, watch):
        need = self._min_cycles(bundle)
        rows = []
        for engine_id, g in data.groupby("engine_id"):
            rul = np.nan
            if len(g) >= need:
                try:
                    rul = float(bundle.history(g.reset_index(drop=True))["RUL"].iloc[-1])
                except Exception:
                    pass
            rows.append({"Engine": int(engine_id), "Cycles given": len(g), "Estimated RUL": rul,
                         "Status": "Not enough data" if np.isnan(rul)
                         else status_for(rul, critical, watch)[0].capitalize()})
        table = pd.DataFrame(rows).sort_values("Estimated RUL", na_position="last")

        counts = table["Status"].value_counts()
        st.subheader(f"{len(table)} engines, closest to failure first")
        st.markdown(f"**{counts.get('Critical', 0)}** critical, **{counts.get('Watch', 0)}** on "
                    f"watch, **{counts.get('Healthy', 0)}** healthy.")
        st.dataframe(table, hide_index=True, column_config={
            "Estimated RUL": st.column_config.ProgressColumn(
                "Estimated RUL (cycles)", min_value=0, max_value=RUL_CAP, format="%d")})
        st.download_button("Download results as CSV", table.round(1).to_csv(index=False).encode(),
                           "rul_predictions.csv", "text/csv")

        usable = table.dropna(subset=["Estimated RUL"])["Engine"].tolist()
        if usable:
            pick = st.selectbox("Look closer at one engine", usable)
            self._show_single(bundle, data[data["engine_id"] == pick].reset_index(drop=True),
                              critical, watch)


def _navigation():
    options = ["Predict", "Analyst dashboard"]
    if hasattr(st, "segmented_control"):
        st.segmented_control("View", options, default=options[0], key="view",
                             label_visibility="collapsed")
    else:
        st.radio("View", options, horizontal=True, key="view", label_visibility="collapsed")


def main():
    st.set_page_config(page_title="Engine Life Predictor", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    assets = get_assets()

    # the choice is kept in session state, so the switch can sit inside the narrow
    # column on the Predict page and line up with the content below it
    view = st.session_state.get("view") or "Predict"
    if view == "Predict":
        _, middle, _ = st.columns([1, 6, 1])
        with middle:
            _navigation()
            PredictPage(assets).run()
    else:
        _navigation()
        ConsoleApp(assets).run()


if __name__ == "__main__":
    main()