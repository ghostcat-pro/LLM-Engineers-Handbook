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
ENV_FILE=.env.local poetry run python -m tools.local training-check
```

For an RTX 3060 12GB, start with QLoRA, sequence length 1024, batch size 1, gradient accumulation 8-16, and LoRA rank 8 or 16.

The current local RAG stack does not require the training packages. Fine-tuning requires a separate dependency pass for `trl`, `peft`, `bitsandbytes`, and `unsloth`.

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
