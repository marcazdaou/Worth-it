import pytest

from app.features import core_title


@pytest.mark.parametrize("title, expected", [
    ("Data Analyst", "data analyst"),
    ("Senior Data Analyst", "data analyst"),
    ("Sr. Data Analyst (Remote)", "data analyst"),
    ("Software Engineer II", "software engineer"),
    ("Junior Product Designer - Hybrid", "product designer"),
    ("Head of Marketing", "marketing"),
    ("", ""),
    (None, ""),
])
def test_core_title_drops_level_and_work_arrangement(title, expected):
    assert core_title(title) == expected


def test_core_title_keeps_words_that_only_contain_level_words():
    # "lead" is removed as a word, but "leadership" and "staffing" are real job words.
    assert core_title("Leadership Development Staffing Coordinator") == "leadership development staffing coordinator"
