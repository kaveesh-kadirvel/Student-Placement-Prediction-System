import pandas as pd

from src.preprocessing import build_preprocessor


def test_preprocessor_transforms_dataset_without_error():
    df = pd.read_csv("train.csv").head(25).drop(columns=["Student_ID", "Placement_Status"])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == len(df)
    assert transformed.size > 0
