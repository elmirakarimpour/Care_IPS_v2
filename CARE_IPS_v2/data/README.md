# Data policy

`raw/CIC-IDS2017/Global test set.csv` is the user-provided held-out test file.
Do not overwrite it. The repository `.gitignore` excludes raw datasets by default;
remove the ignore rule only if there is an explicit reason and permission to publish
the dataset (check the dataset license/terms first).

Place a separate training dataset under `raw/CIC-IDS2017/train/`.
Do not use the global test set to fit a scaler, select features, tune hyperparameters,
calibrate probabilities, or choose thresholds.
