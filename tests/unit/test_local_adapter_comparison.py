import json
from pathlib import Path

from click.testing import CliRunner

from tools.local import _load_prompts_from_jsonl, compare_adapter_command


def test_compare_adapter_command_writes_report(monkeypatch, tmp_path: Path) -> None:
    output_file = tmp_path / "comparison.json"

    def fake_generate_local_response(prompt, config, *, adapter_dir, max_new_tokens, temperature):
        suffix = "adapter" if adapter_dir is not None else "base"
        return f"{prompt}::{suffix}"

    monkeypatch.setattr(
        "llm_engineering.model.finetuning.local.generate_local_response",
        fake_generate_local_response,
    )

    result = CliRunner().invoke(
        compare_adapter_command,
        [
            "--prompt",
            "Explain SFT.",
            "--output-file",
            str(output_file),
            "--max-new-tokens",
            "8",
            "--temperature",
            "0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_file.read_text())
    assert payload["results"] == [
        {
            "prompt": "Explain SFT.",
            "base_response": "Explain SFT.::base",
            "adapter_response": "Explain SFT.::adapter",
            "identical": False,
        }
    ]


def test_load_prompts_from_jsonl_reads_instruction_or_prompt(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "Explain SFT."}),
                json.dumps({"prompt": "Explain LoRA."}),
                json.dumps({"output": "ignored"}),
            ]
        )
        + "\n"
    )

    assert _load_prompts_from_jsonl(prompts_file, max_prompts=2) == ["Explain SFT.", "Explain LoRA."]


def test_compare_adapter_command_reads_prompts_file(monkeypatch, tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "comparison.json"
    prompts_file.write_text(json.dumps({"instruction": "Explain SFT."}) + "\n")

    def fake_generate_local_response(prompt, config, *, adapter_dir, max_new_tokens, temperature):
        suffix = "adapter" if adapter_dir is not None else "base"
        return f"{prompt}::{suffix}"

    monkeypatch.setattr(
        "llm_engineering.model.finetuning.local.generate_local_response",
        fake_generate_local_response,
    )

    result = CliRunner().invoke(
        compare_adapter_command,
        [
            "--prompts-file",
            str(prompts_file),
            "--output-file",
            str(output_file),
            "--max-new-tokens",
            "8",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_file.read_text())
    assert payload["results"][0]["prompt"] == "Explain SFT."
