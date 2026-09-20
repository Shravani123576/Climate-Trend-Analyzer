#from pathlib import Path
#from src.data_loader import load_raw_data

#def test_load_data():
    #dataset = Path("data/raw/GLB.Ts+dSST.csv")
    #df = load_raw_data(dataset)
    #assert not df.empty 
from pathlib import Path

from src.data_loader import (
    load_raw_data,
    dataset_summary,
    validate_dataset,
    preview_data,
)


DATASET = (
    Path("data")
    / "raw"
    / "GLB.Ts+dSST.csv"
)


def test_load_raw_data():
    df = load_raw_data(DATASET)

    assert not df.empty
    assert "Year" in df.columns
    assert "J-D" in df.columns


def test_dataset_summary():
    df = load_raw_data(DATASET)

    summary = dataset_summary(df)

    assert summary["Rows"] > 0
    assert summary["Columns"] == 19


def test_validate_dataset():
    df = load_raw_data(DATASET)

    assert validate_dataset(df) is True


def test_preview_data():
    df = load_raw_data(DATASET)

    preview = preview_data(df, rows=5)

    assert len(preview) == 5