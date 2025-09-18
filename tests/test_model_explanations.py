import pandas as pd
from procurement_risk_detection_ai.models.baseline_model import (
    fit_baseline,
    predict_proba_and_contrib,
    FEATURE_COLS_DEFAULT,
)
from sklearn.linear_model import LogisticRegression


def test_predict_proba_and_contrib_linear_explanations():
    # synthetic training data
    df = pd.DataFrame(
        {
            "award_concentration_by_buyer": [0.1, 0.9, 0.8, 0.2],
            "repeat_winner_ratio": [0.0, 0.95, 0.8, 0.1],
            "amount_zscore_by_category": [0.2, 3.0, 2.5, 0.1],
            "near_threshold_flag": [0, 1, 1, 0],
            "time_to_award_days": [10, 120, 90, 15],
        }
    )
    clf = fit_baseline(df, FEATURE_COLS_DEFAULT)
    assert isinstance(clf, LogisticRegression)

    # score one row
    row = df.iloc[1]
    prob, contribs = predict_proba_and_contrib(clf, FEATURE_COLS_DEFAULT, row)
    assert 0.0 <= prob <= 1.0
    assert len(contribs) == len(FEATURE_COLS_DEFAULT)
    # biggest contributor should be around large zscore / repeat_winner / near_threshold
    names_sorted = [c[0] for c in contribs]
    assert (
        "repeat_winner_ratio" in names_sorted[:3]
        or "amount_zscore_by_category" in names_sorted[:3]
    )
