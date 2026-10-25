# Engine Remaining Useful Life Predictor

A small project that estimates how many flight cycles a jet engine has left before it needs maintenance. It is trained on NASA's C-MAPSS turbofan data (the FD001 subset) and comes with a Streamlit app where you can upload your own sensor readings and get an answer.

I built it to compare two very different approaches on the same problem: gradient boosted trees on hand-made rolling features (XGBoost), and a recurrent network that reads a window of recent cycles (LSTM). The app lets you use either one.

## What the app does:

The app has two views, switched from the top of the page.

**Predict** is the main page. You give it sensor readings in one of three ways:

1. Upload a CSV file.
2. Type readings into an editable table.
3. Try one of the real test engines from the NASA data. In this mode the page also shows the true remaining life next to the model's guess, so you can see how close it got.

The result is a gauge, a large number of remaining flight cycles, a status (healthy, watch or critical) and a one line suggestion. If your file contains several engines, you get a table sorted by how close each one is to failure, and you can download it as a CSV.

**Analyst dashboard** is the longer view for people who want to dig into the models. It has four tabs:

- Live replay: pick a test engine and watch the predictions evolve cycle by cycle against the true curve.
- Fault injection: make one sensor drift away from normal over the last few cycles and see how much the estimate moves.
- Fleet view: scores for all 100 test engines, a scatter of predicted against actual, and a maintenance queue.
- Under the hood: the class hierarchy, which methods each class overrides, and a small encapsulation demo.

## Project layout
```
predictive_maintenance/
│
├── .streamlit/
│   └── config.toml             # Streamlit theme and app configuration
│
├── CMaps/                      # NASA C-MAPSS dataset files
│
├── src/
│   ├── __init__.py
│   ├── base.py                 # Abstract classes, metrics, and experiment runner
│   ├── handle_data.py          # Dataset loading and RUL calculation
│   └── models.py               # Preprocessing and XGBoost/LSTM model definitions
│
├── xgboost.ipynb           # XGBoost training and evaluation
├── lstm_model.ipynb       # LSTM training and evaluation
│
├── xgb_model.json          # Saved XGBoost model
├── lstm_model.keras        # Saved LSTM model
│
├── app.py                      # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Files excluded from version control
```

The `.streamlit` folder starts with a dot, so Finder and some file explorers hide it. It has to sit in the project root, next to `app.py`, and not inside `src`.

## Setup

You need Python and a few libraries:

```
pip install streamlit pandas numpy scikit-learn xgboost tensorflow altair
```

Put the C-MAPSS text files (`train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt`) in the `CMaps` folder. The dataset is published by the NASA Prognostics Center of Excellence.

## Running it

From the project root:

```
streamlit run app.py
```

The app opens in your browser. The first start takes a little while because it loads the data and prepares the models.

Two details about the model files:

- If `xgb_model.json` is missing, the app trains XGBoost on the spot and saves the file.
- If `lstm_model.keras` is missing, the app runs with XGBoost only. Run `lstm_model.ipynb` to create it.

The theme is read from `.streamlit/config.toml` when Streamlit starts, so restart the app after you change that file.

## Using your own data

Your file needs one row per flight cycle and these columns:

- `setting_1` and `setting_2`
- `sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`

Two more columns are optional. `cycle` is the flight number counted from the engine's first flight (if it is missing, rows are numbered 1, 2, 3 in the order given). `engine_id` lets you score several engines in one file.

Commas, semicolons and tabs all work as separators, and so do the original NASA `.txt` files that have no header row. The app has a button that downloads a template CSV with the right layout. Blank cells are filled with the nearest reading from the same engine, and you are told when that happens.

Try to give at least 30 cycles per engine. XGBoost will run on less, but the estimate gets shaky, and the LSTM refuses to run on fewer than 30.

## How the models work

All the models share the same preparation steps:

- Seven columns that never change in FD001 are dropped, which leaves 15 informative sensors and two operating settings.
- The training target is capped at 125 cycles. An engine that is far from failure looks the same as one that is somewhat far from failure, so asking the model to tell them apart only adds noise.
- For every sensor, a rolling mean and rolling standard deviation over a 27 cycle window are added as features.

**XGBoost** gets one row per cycle with those features. Predictions are clipped to the range 0 to 125.

**LSTM** gets sliding windows of 30 cycles, scaled to the 0 to 1 range with a scaler that is fitted on the training data only. It is a single LSTM layer followed by one output unit, trained with early stopping and a learning rate that drops when progress stalls.

Both are scored with MAE, MSE, RMSE and R2, plus the asymmetric NASA scoring function, which punishes late predictions harder than early ones.

## Code structure

I used this project to practise object oriented design, and the code is organised around it:

- `BaseRULModel` and `BaseRULPreprocessor` in `base.py` are abstract classes. They define what every model and preprocessor must provide.
- `XGBRULModel`, `LSTMRULModel`, `TabularPreprocessor` and `SequencePreprocessor` in `models.py` inherit from them.
- `ExperimentRunner` and the app only talk to the abstract interfaces, so adding a third model means writing one new model class and one new preprocessor, without touching the app logic.
- Settings such as `rul_cap` are private and go through validating properties.

The "Under the hood" tab in the dashboard shows all of this live.

## Limitations

- The models were trained on FD001, which has a single operating condition and a single fault type. They will not be reliable for engines that behave differently.
- The `cycle` number is one of the XGBoost features, so your cycle counting should start from the engine's first flight.
- Estimates stop at 125 cycles by design. For a healthy engine the app shows "125+".
- The LSTM scores in the notebook use the uncapped true value, while the dashboard caps it for every model so the comparison is fair. The numbers will differ a little.

## License

See the `LICENSE` file in the project root.
