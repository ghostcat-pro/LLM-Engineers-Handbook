# Local Training Preparation

This phase prepares a local-only fine-tuning path for the RTX 3060 12GB machine.

It does not use AWS SageMaker, Hugging Face Hub uploads, OpenAI, Comet, or managed services.

## Current Status

The original project training path is cloud-first:

- ZenML pipeline orchestration
- SageMaker training job
- Hugging Face datasets and model upload
- Comet reporting

The local-first path is separate and conservative. The first milestone is to prepare local JSONL datasets and a safe QLoRA profile before installing GPU training dependencies.

## RTX 3060 12GB Profile

Use the local config:

```bash
configs/local_training_rtx3060.yaml
```

Initial profile:

- base model: `models/mistral-7b`
- quantization: 4-bit
- max sequence length: 1024
- batch size: 1
- gradient accumulation: 16
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- epochs: 1
- reporting: local/off
- Hub upload: disabled

This is intentionally smaller than the book's SageMaker profile. The goal is to avoid VRAM failures first, then increase carefully.

Ollama models are fine for inference, but they are not the same thing as trainable Hugging Face-format weights. The local trainer refuses to download model weights. Use a local model directory such as `models/mistral-7b`, or pre-populate the Hugging Face cache before training.

## Inspect The Plan

```bash
poetry run poe local-training-plan
```

Expected result: prints the local RTX 3060 training profile from `.env.local`.

## Prepare Local Training Data

First generate local datasets if they do not exist:

```bash
ENV_FILE=.env.local poetry run python -m tools.local generate-dataset --dataset-type instruction --max-prompts 1
ENV_FILE=.env.local poetry run python -m tools.local generate-dataset --dataset-type preference --max-prompts 1
```

Then convert them to local JSONL files:

```bash
poetry run poe local-prepare-training-data
```

Expected outputs:

- `data/training/datasets/sft_train.jsonl`
- `data/training/datasets/sft_test.jsonl`
- `data/training/datasets/dpo_train.jsonl`
- `data/training/datasets/dpo_test.jsonl`

These files are ignored by git.

## Prepare Thesis Training Data

For the thesis/domain experiment, generate thesis-weighted samples first:

```bash
poetry run poe local-generate-thesis-dataset
```

Inspect the generated file:

```text
data/generated/local_thesis_sft_dataset.jsonl
```

Then prepare the trainer split:

```bash
poetry run poe local-prepare-thesis-training-data
```

Expected outputs:

- `data/training/datasets/sft_train.jsonl`
- `data/training/datasets/sft_test.jsonl`

This command does not call external services. It reads the local JSONL, keeps valid `instruction`/`output` rows, preserves source metadata, and writes a deterministic train/eval split for the SFT trainer.

## Check Dependencies

```bash
poetry run poe local-training-check
```

Expected current result: `trl`, `peft`, and `bitsandbytes` are installed. `unsloth` may still be missing, which is fine because the guarded local trainer does not require it.

Current local SFT dependency set:

- `accelerate==0.33.0`
- `trl==0.10.1`
- `peft==0.12.0`
- `bitsandbytes==0.43.3`

`unsloth` is not required for the guarded local trainer.

## Dry Run

```bash
poetry run poe local-train-dry-run
```

Purpose: validates local training readiness without loading the model.

It checks:

- local-first flags
- no Hub upload
- CUDA visibility
- training dependencies
- local/cached model availability
- JSONL training data
- RTX 3060-safe batch/sequence/LoRA settings

## Actual SFT Command

```bash
poetry run poe local-train-sft
```

This command is intentionally explicit. It reads JSONL files from `data/training/datasets`, writes adapters/checkpoints under `data/training/runs`, reports locally only, and never pushes to the Hub.

## Local Adapter Inference

Run the base model only:

```bash
poetry run poe local-infer-base
```

Run the base model with the saved LoRA adapter:

```bash
poetry run poe local-infer-adapter
```

The default adapter path is:

```text
data/training/runs/mistral-7b-local-qlora/adapter
```

To use a custom prompt:

```bash
ENV_FILE=.env.local poetry run python -m tools.local infer --prompt "Explain LoRA in one paragraph."
```

## Compare Base And Adapter

```bash
poetry run poe local-compare-adapter
```

Purpose: runs the same prompts through the base model and the saved LoRA adapter, then writes a local JSON report under:

```text
data/training/evaluations/
```

With the current one-sample training run, identical or near-identical responses are expected. This command becomes more useful after generating a larger local dataset and retraining.

Compare using prompts from the prepared local SFT JSONL:

```bash
poetry run poe local-compare-adapter-sft
```

You can also provide any JSONL file containing `instruction` or `prompt` fields:

```bash
ENV_FILE=.env.local poetry run python -m tools.local compare-adapter --prompts-file data/training/datasets/sft_train.jsonl --max-prompts 3
```
