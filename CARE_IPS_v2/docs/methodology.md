# Methodology and data leakage controls

1. Keep the global test file untouched.
2. Split training data into train/validation before fitting preprocessing.
3. Fit missing-value handling, constant/redundant feature removal, scaling,
   feature selection, and calibration on train only.
4. Apply the fitted transformations to validation and global test.
5. Tune thresholds/model settings on validation only.
6. Evaluate once on the held-out global test and preserve per-row predictions.

The uploaded Global Test Set has 15,999 rows and 78 columns. Its observed labels
are BENIGN (15,805) and FTP-Patator (194). It is therefore a two-class,
strongly imbalanced test sample, not a substitute for the complete CIC-IDS2017
training corpus. Report macro-F1, per-class recall/precision, confusion matrix,
and calibration measures; do not rely on accuracy alone.
