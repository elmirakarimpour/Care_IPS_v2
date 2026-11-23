from pathlib import Path
def test_global_test_file_is_present():
    assert Path("data/raw/CIC-IDS2017/Global test set.csv").exists()
