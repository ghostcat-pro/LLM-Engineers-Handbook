from pathlib import Path

from llm_engineering.model.finetuning import local as local_training
from llm_engineering.model.finetuning.local import (
    LocalTrainingConfig,
    _default_adapter_dir,
    _format_sft_prompt,
    check_local_training_readiness,
)
from tools.local import _flatten_dataset_samples, _write_jsonl


def test_flatten_dataset_samples_collects_split_samples() -> None:
    payload = {
        "train": {
            "articles": {
                "samples": [
                    {"instruction": "a", "answer": "b"},
                    {"instruction": "c", "answer": "d"},
                ]
            },
            "posts": {"samples": [{"instruction": "e", "answer": "f"}]},
        },
        "test": {"articles": {"samples": []}},
    }

    samples = _flatten_dataset_samples(payload, "train")

    assert samples == [
        {"instruction": "a", "answer": "b"},
        {"instruction": "c", "answer": "d"},
        {"instruction": "e", "answer": "f"},
    ]


def test_write_jsonl_handles_empty_and_non_empty_records(tmp_path: Path) -> None:
    populated = tmp_path / "records.jsonl"
    empty = tmp_path / "empty.jsonl"

    _write_jsonl(populated, [{"a": 1}, {"b": 2}])
    _write_jsonl(empty, [])

    assert populated.read_text().splitlines() == ['{"a": 1}', '{"b": 2}']
    assert empty.read_text() == ""


def test_local_training_readiness_accepts_local_model_and_data(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    data_dir = tmp_path / "data"
    model_dir.mkdir()
    _write_jsonl(data_dir / "sft_train.jsonl", [{"instruction": "a", "output": "b"}])
    _write_jsonl(data_dir / "sft_test.jsonl", [])

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(index: int) -> str:
            return "NVIDIA GeForce RTX 3060"

    class Torch:
        cuda = Cuda()

    config = LocalTrainingConfig(
        stage="sft",
        base_model=str(model_dir),
        output_dir=tmp_path / "runs",
        data_dir=data_dir,
        max_seq_length=1024,
        load_in_4bit=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_epochs=1,
    )
    monkeypatch.setattr(local_training, "_package_available", lambda package: True)
    monkeypatch.setitem(__import__("sys").modules, "torch", Torch)

    readiness = check_local_training_readiness(config)

    assert readiness.ok
    assert not readiness.failures


def test_local_training_readiness_rejects_uncached_remote_model(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "sft_train.jsonl", [{"instruction": "a", "output": "b"}])
    _write_jsonl(data_dir / "sft_test.jsonl", [])

    config = LocalTrainingConfig(
        stage="sft",
        base_model="remote/model",
        output_dir=tmp_path / "runs",
        data_dir=data_dir,
        max_seq_length=1024,
        load_in_4bit=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_epochs=1,
    )
    monkeypatch.setattr(local_training, "_package_available", lambda package: True)

    readiness = check_local_training_readiness(config)

    assert not readiness.ok
    assert any("Base model is not local/cached" in failure for failure in readiness.failures)


def test_format_sft_prompt_matches_training_template() -> None:
    prompt = _format_sft_prompt("Explain SFT.")

    assert "### Instruction:\nExplain SFT." in prompt
    assert prompt.endswith("### Response:\n")


def test_default_adapter_dir_uses_output_dir(tmp_path: Path) -> None:
    config = LocalTrainingConfig(
        stage="sft",
        base_model=str(tmp_path / "model"),
        output_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        max_seq_length=1024,
        load_in_4bit=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_epochs=1,
    )

    assert _default_adapter_dir(config) == tmp_path / "runs" / "adapter"
