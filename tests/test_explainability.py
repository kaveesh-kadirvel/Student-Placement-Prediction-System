import pandas as pd
import pytest

from src.explainability import explain_model, explain_single_prediction, shap
from src.predict import load_model
from src.schema import FEATURE_COLUMNS


@pytest.mark.skipif(shap is None, reason="SHAP is unavailable")
def test_shap_global_and_individual_outputs_match_transformed_schema():
    model = load_model("placement_model.pkl")
    sample = pd.read_csv("train.csv").loc[:7, FEATURE_COLUMNS]

    global_result = explain_model(model, sample)
    local_result = explain_single_prediction(model, sample.iloc[[0]])
    predicted_class = model.predict(sample.iloc[[0]])[0]

    assert global_result["status"] == "available"
    assert global_result["explained_rows"] == len(sample)
    assert len(global_result["feature_names"]) == len(global_result["importance"])
    assert local_result["status"] == "available"
    assert local_result["top_features"]
    assert all("contribution" in item and "feature" in item for item in local_result["top_features"])
    transformed_names = set(model.named_steps["preprocessor"].get_feature_names_out())
    assert all(item["feature"] in transformed_names for item in local_result["top_features"])
    assert any(item["contribution"] != 0 for item in local_result["top_features"])
    assert local_result["explained_class"] == predicted_class

    candidates = pd.read_csv("train.csv").loc[100:200, FEATURE_COLUMNS]
    negative_row = candidates.loc[model.predict(candidates) == 0].iloc[[0]]
    negative_result = explain_single_prediction(
        model, negative_row, background_df=sample
    )
    assert negative_result["explained_class"] == 0
