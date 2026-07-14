from app.knowledge.loader import load_knowledge


def test_load_default_knowledge_contains_core_rules():
    text = load_knowledge()
    assert "형" in text
    assert "눈치" in text
    assert "어이가 없네" in text


def test_load_knowledge_from_custom_dir(tmp_path):
    (tmp_path / "custom.yaml").write_text(
        "rules:\n  - term: 테스트어\n    rule: 테스트 규칙\n    bad: bad ex\n    good: good ex\n",
        encoding="utf-8",
    )
    text = load_knowledge(str(tmp_path))
    assert "테스트어" in text
    assert "good ex" in text
