from src.app.api.main import demo_score

import sys

# adding Folder_2 to the system path
sys.path.insert(0, "src")


def test_demo_score_monotonic():
    # Sanctioned should increase score strongly
    base = demo_score(0, 0, False, 0)
    sanc = demo_score(0, 0, True, 0)
    assert sanc > base
