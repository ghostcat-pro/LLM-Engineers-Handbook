import json
from pathlib import Path

from click.testing import CliRunner

from tools import local
from tools.local import _load_jsonl, _score_expected_points, _write_jsonl, evaluate_thesis_rag_command


def test_score_expected_points_matches_phrases_and_tokens() -> None:
    answer = "The final binary XGBoost model achieved macro-F1 of 0.77 and ROC-AUC of 0.89."

    score = _score_expected_points(
        answer,
        [
            "binary XGBoost model",
            "macro-F1 of 0.77",
            "ROC-AUC of 0.89",
            "top 10% ranking",
        ],
    )

    assert score["coverage"] == 0.75
    assert score["covered"] == [
        "binary XGBoost model",
        "macro-F1 of 0.77",
        "ROC-AUC of 0.89",
    ]
    assert score["missing"] == ["top 10% ranking"]


def test_score_expected_points_accepts_close_paraphrase() -> None:
    answer = (
        "The thesis formalizes and predicts observable managerial adjustment patterns from historical "
        "scheduling data to support decision-support and preemptive repair workflows."
    )

    score = _score_expected_points(
        answer,
        [
            "predict managerial adjustment patterns",
            "historical scheduling data",
            "decision-support and preemptive repair workflows",
        ],
    )

    assert score["coverage"] == 1.0


def test_load_jsonl_rejects_non_object_records(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[]\n")

    try:
        _load_jsonl(path)
    except ValueError as exc:
        assert "must contain a JSON object" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_evaluate_thesis_rag_command_writes_local_report(monkeypatch, tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.jsonl"
    output_file = tmp_path / "report.json"
    _write_jsonl(
        eval_file,
        [
            {
                "id": "objective_001",
                "category": "objective",
                "question": "What is the thesis objective?",
                "expected_points": ["predict managerial adjustments", "workforce scheduling"],
                "expected_source": "thesis.pdf",
                "required_chunk_indices": [7],
                "retrieval_queries": ["managerial adjustments"],
            }
        ],
    )

    class FakeProvider:
        def generate(self, prompt: str, **kwargs: object) -> str:
            self.prompt = prompt
            return "The thesis aims to predict managerial adjustments in workforce scheduling."

    def fake_get_llm_provider() -> FakeProvider:
        return FakeProvider()

    search_calls = []

    def fake_search_local_sources(query: str, limit: int, **kwargs: object) -> list[dict]:
        search_calls.append({"query": query, **kwargs})
        return [
            {
                "source_path": "data/local_sources/thesis.pdf",
                "source_name": "thesis.pdf",
                "chunk_index": 1,
                "score": 0.9,
                "content": "The thesis predicts managerial adjustments in workforce scheduling.",
            }
        ]

    monkeypatch.setattr("llm_engineering.application.llm.get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr("llm_engineering.application.local_sources.search_local_sources", fake_search_local_sources)
    monkeypatch.setattr(local.settings, "USE_MLFLOW", False)

    result = CliRunner().invoke(
        evaluate_thesis_rag_command,
        [
            "--eval-file",
            str(eval_file),
            "--output-file",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_file.read_text())
    assert payload["aggregate"]["num_cases"] == 1
    assert payload["aggregate"]["avg_expected_point_coverage"] == 1.0
    assert payload["aggregate"]["expected_source_retrieval_rate"] == 1.0
    assert payload["results"][0]["expected_point_score"]["missing"] == []
    assert any(call.get("source_name") == "thesis.pdf" for call in search_calls)
    assert any(call.get("chunk_index") == 7 for call in search_calls)
