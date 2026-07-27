# Local-First LLM Twin

This profile runs the LLM Twin experiment without AWS, OpenAI, Hugging Face Hub, Comet/Opik, or managed cloud services.

## Runtime

- MongoDB: Docker Compose
- Qdrant: Docker Compose
- LLM: Ollama on localhost
- Default chat model: `qwen2.5:7b-instruct`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Metrics: local MLflow file store under `data/mlruns`

## Hardware Note

The target machine has an RTX 3060 12GB. If `nvidia-smi` works in your normal terminal, host GPU support is available. The Codex sandbox may still report that CUDA is not visible because it cannot access `/dev/nvidia*`; in that case, run GPU training commands from your normal terminal and use Codex to adjust code/configs from logs.

## First Run

Start local databases:

```bash
poetry run poe local-stack-up
```

Start Ollama on the host:

```bash
ollama serve
```

Confirm the model exists:

```bash
ollama list
```

Validate the local-first configuration:

```bash
poetry run poe local-validate
```

Local health check:

```bash
ENV_FILE=.env.local poetry run python -m tools.local healthcheck
```

Run the smoke test:

```bash
poetry run poe local-smoke-test
```

The smoke test checks local services, Ollama generation, Qdrant retrieval, and the FastAPI RAG endpoint. For a slower check that also generates a tiny dataset and local evaluation report:

```bash
poetry run poe local-smoke-test-full
```

Run fast local unit tests without live services:

```bash
poetry run poe local-test
```

Import bundled data and build vectors:

```bash
ENV_FILE=.env.local poetry run python -m tools.local import-data
ENV_FILE=.env.local poetry run python -m tools.local build-vector-db --reset
```

Run the API:

```bash
poetry run poe local-run-api
```

Test RAG:

```bash
poetry run poe local-rag-test
```

Evaluate the local thesis RAG baseline:

```bash
poetry run poe local-evaluate-thesis-rag
```

Reports are written under `data/evaluations/`. The tracked evaluation cases live in `data/evaluation/thesis_eval_questions.jsonl`.

## Local Dataset Generation

Generate a small instruction dataset:

```bash
ENV_FILE=.env.local poetry run python -m tools.local generate-dataset --dataset-type instruction --max-prompts 1
```

Generate a small preference dataset:

```bash
ENV_FILE=.env.local poetry run python -m tools.local generate-dataset --dataset-type preference --max-prompts 1
```

Outputs are written to `data/generated/`.

The default examples use one prompt so you can verify the path quickly. Increase `--max-prompts` for longer local generation runs.

## Local Evaluation

Run a local RAG evaluation:

```bash
ENV_FILE=.env.local poetry run python -m tools.local evaluate-rag
```

Add `--judge` to ask the local model for JSON scores.

Outputs are written to `data/evaluations/` and aggregate metrics are logged to local MLflow when enabled.

To inspect local MLflow output later, point MLflow at the file store:

```bash
mlflow ui --backend-store-uri data/mlruns
```

## Training Readiness

Check local fine-tuning readiness:

```bash
poetry run poe local-training-check
```

Inspect the conservative RTX 3060 profile:

```bash
poetry run poe local-training-plan
```

Prepare generated datasets for local training:

```bash
poetry run poe local-prepare-training-data
```

Validate the local trainer without loading the model:

```bash
poetry run poe local-train-dry-run
```

For an RTX 3060 12GB, start with QLoRA, sequence length 1024, batch size 1, gradient accumulation 8-16, and LoRA rank 8 or 16.

The local SFT path uses `trl`, `peft`, and `bitsandbytes`. `unsloth` is still optional and is not required by the guarded local trainer.

See [LOCAL_TRAINING.md](/home/fspinto/projects/LLM-Engineers-Handbook/LOCAL_TRAINING.md) for the local training preparation notes.

## Thesis And Local Sources

Place thesis/articles under:

```text
data/local_sources/
```

Then import and test retrieval:

```bash
poetry run poe local-import-sources
poetry run poe local-search-sources
```

See [LOCAL_SOURCES.md](/home/fspinto/projects/LLM-Engineers-Handbook/LOCAL_SOURCES.md) for the thesis/source workflow.

## Local-First Guarantees

With `.env.local`, these are disabled by default:

- AWS/SageMaker
- OpenAI
- Hugging Face Hub uploads
- Comet/Opik cloud logging
- ZenML secret store loading

The legacy `tools.run` ZenML entry point is guarded while `USE_CLOUD=false`, so the original cloud/SageMaker pipeline commands fail fast instead of accidentally leaving the local-first profile.

Generated local run outputs are ignored by git under:

- `data/generated/`
- `data/evaluations/`
- `data/mlruns/`
