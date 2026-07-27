from pathlib import Path

from llm_engineering.model.finetuning import local as local_training
from llm_engineering.model.finetuning.local import (
    LocalTrainingConfig,
    _default_adapter_dir,
    _format_sft_prompt,
    check_local_training_readiness,
)
from tools.local import (
    _flatten_dataset_samples,
    _load_sft_samples_from_jsonl,
    _split_train_eval,
    _write_jsonl,
    prepare_thesis_training_data_command,
)


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


def test_load_sft_samples_from_jsonl_keeps_valid_local_metadata(tmp_path: Path) -> None:
    path = tmp_path / "thesis.jsonl"
    _write_jsonl(
        path,
        [
            {
                "instruction": " What is the thesis about? ",
                "output": " Managerial adjustments. ",
                "source_name": "thesis.pdf",
                "source_path": "data/local_sources/thesis.pdf",
                "chunk_index": 3,
            },
            {"instruction": "ignored"},
            {"instruction": "", "output": "ignored"},
        ],
    )

    assert _load_sft_samples_from_jsonl(path) == [
        {
            "instruction": "What is the thesis about?",
            "output": "Managerial adjustments.",
            "source_name": "thesis.pdf",
            "source_path": "data/local_sources/thesis.pdf",
            "chunk_index": 3,
        }
    ]


def test_split_train_eval_is_deterministic_and_keeps_train_sample() -> None:
    samples = [{"instruction": str(index), "output": str(index)} for index in range(10)]

    train_samples, eval_samples = _split_train_eval(samples, test_size=0.2)

    assert train_samples == samples[:8]
    assert eval_samples == samples[8:]


def test_split_train_eval_handles_single_sample() -> None:
    samples = [{"instruction": "a", "output": "b"}]

    assert _split_train_eval(samples, test_size=0.1) == (samples, [])


def test_prepare_thesis_training_data_command_writes_sft_split(tmp_path: Path) -> None:
    from click.testing import CliRunner

    input_file = tmp_path / "local_thesis_sft_dataset.jsonl"
    output_dir = tmp_path / "datasets"
    _write_jsonl(
        input_file,
        [
            {"instruction": "a", "output": "b"},
            {"instruction": "c", "output": "d"},
            {"instruction": "e", "output": "f"},
        ],
    )

    result = CliRunner().invoke(
        prepare_thesis_training_data_command,
        [
            "--input-file",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--test-size",
            "0.34",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "sft_train.jsonl").read_text().splitlines() == [
        '{"instruction": "a", "output": "b"}',
        '{"instruction": "c", "output": "d"}',
    ]
    assert (output_dir / "sft_test.jsonl").read_text().splitlines() == ['{"instruction": "e", "output": "f"}']


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
