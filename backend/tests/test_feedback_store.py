from app.feedback.store import FeedbackStore
from app.schemas import FeedbackEntry


def entry(action="approved", final=""):
    return FeedbackEntry(
        movie="테스트영화", segment_id="pair_1", korean="어이가 없네",
        dubbed="I have no kidney", finding_id="f1",
        reviewer_action=action, final_text=final,
    )


def test_record_appends_jsonl_with_timestamp(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.jsonl"))
    store.record(entry())
    store.record(entry(action="modified", final="This is ridiculous."))
    rows = store.all()
    assert len(rows) == 2
    assert rows[0]["reviewer_action"] == "approved"
    assert rows[0]["timestamp"] != ""
    assert rows[1]["final_text"] == "This is ridiculous."


def test_store_creates_parent_dir(tmp_path):
    store = FeedbackStore(str(tmp_path / "nested" / "fb.jsonl"))
    store.record(entry())
    assert len(store.all()) == 1


def test_all_returns_empty_for_missing_file(tmp_path):
    store = FeedbackStore(str(tmp_path / "none.jsonl"))
    assert store.all() == []
