from procurement_risk_detection_ai.app.api.main import demo_score


def test_demo_score_monotonic():
    # Sanctioned should increase score strongly
    base = demo_score(0, 0, False, 0)
    sanc = demo_score(0, 0, True, 0)
    assert sanc > base
