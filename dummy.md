# Local-First Experiment Runbook

This document is a step-by-step sequence to run the local LLM Twin experiment.

The goal is to run everything locally:

- MongoDB for the local document warehouse
- Qdrant for the local vector database
- Ollama with `qwen2.5:7b-instruct` for local generation
- local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- FastAPI for the RAG endpoint
- local JSON and MLflow files for generated/evaluation artifacts

No AWS, OpenAI, Hugging Face Hub upload, Comet/Opik cloud logging, or managed cloud service is required.

## 0. Open Two Terminals

Use one terminal for Ollama and another terminal for project commands.

In all project commands, run from the repository root:

```bash
cd /home/fspinto/projects/LLM-Engineers-Handbook
```

Expected result: your prompt is inside the project directory.

## 1. Start Ollama

In terminal 1:

```bash
ollama serve
```

Purpose: starts the local LLM server on `http://localhost:11434`.

Expected result: Ollama stays running and waits for requests.

If it says the port is already in use, Ollama is probably already running. That is fine.

## 2. Confirm The Local Model Exists

In terminal 2:

```bash
ollama list
```

Purpose: confirms Ollama has the model used by this project.

Expected result: the list includes:

```text
qwen2.5:7b-instruct
```

If it is missing, pull it:

```bash
ollama pull qwen2.5:7b-instruct
```

## 3. Start Local Databases

In terminal 2:

```bash
poetry run poe local-stack-up
```

Purpose: starts MongoDB and Qdrant using Docker Compose.

Expected result: Docker starts:

- `llm_engineering_mongo`
- `llm_engineering_qdrant`

If Docker reports an NVIDIA runtime error for these database containers, check that `docker-compose.yml` contains `runtime: runc` for both services.

## 4. Validate Local-First Configuration

```bash
poetry run poe local-validate
```

Purpose: checks that the active `.env.local` profile is configured for local-only execution.

Expected result:

```text
Local-first validation passed.
```

This command also reports legacy cloud references that still exist in the original project files. Those references are acceptable when they are behind local guards or old cloud-only commands.

## 5. Check Local Services

```bash
ENV_FILE=.env.local poetry run python -m tools.local healthcheck
```

Purpose: checks whether MongoDB, Qdrant, and Ollama are reachable.

Expected result:

```text
[OK] MongoDB: ...
[OK] Qdrant: ...
[OK] Ollama: ...
```

If MongoDB or Qdrant fails, run:

```bash
poetry run poe local-stack-up
```

If Ollama fails, make sure `ollama serve` is running.

## 6. Check Hardware Visibility

```bash
ENV_FILE=.env.local poetry run python -m tools.local check-hardware
```

Purpose: prints CPU/RAM/GPU visibility from the current process.

Expected result on this machine:

- PCI GPU should show RTX 3060 / GA106
- Your normal terminal should have working `nvidia-smi`
- Codex may still report CUDA as not visible because the sandbox cannot access `/dev/nvidia*`

This is acceptable for RAG and Ollama inference. For future GPU training, run training commands from the normal terminal where `nvidia-smi` works.

## 7. Import Bundled Data Into MongoDB

```bash
ENV_FILE=.env.local poetry run python -m tools.local import-data
```

Purpose: imports the local JSON files from `data/data_warehouse_raw_data` into MongoDB.

Expected result:

- article documents are imported into `articles`
- user documents are imported into `users`
- empty post/repository JSON files are reported as empty

If the collections already contain documents, the command skips them. To reimport from scratch:

```bash
ENV_FILE=.env.local poetry run python -m tools.local import-data --reset
```

## 8. Build The Local Vector Database

```bash
ENV_FILE=.env.local poetry run python -m tools.local build-vector-db --reset
```

Purpose:

1. reads raw documents from MongoDB
2. cleans the documents
3. chunks them
4. embeds chunks locally
5. loads vectors into Qdrant

Expected result:

- Qdrant collections are created
- `embedded_articles` receives vectors
- with the bundled data, around 500 article vectors are expected

This can take a little while because it loads the embedding model.

## 9. Run The Quick Smoke Test

```bash
poetry run poe local-smoke-test
```

Purpose: checks the core local experiment path.

It verifies:

- MongoDB
- Qdrant
- Ollama
- direct local LLM generation
- RAG retrieval
- FastAPI `/rag` endpoint

Expected result:

```text
Local-first smoke test passed.
```

If this passes, the core local RAG system is working.

## 10. Run The Full Smoke Test

```bash
poetry run poe local-smoke-test-full
```

Purpose: checks the full local non-training workflow.

It verifies everything from the quick smoke test, plus:

- tiny local instruction dataset generation
- local RAG evaluation
- local MLflow metric logging

Expected result:

```text
Local-first smoke test passed.
```

Generated files are written under:

- `data/generated/`
- `data/evaluations/`
- `data/mlruns/`

These folders are ignored by git.

## 11. Run The FastAPI Service

```bash
poetry run poe local-run-api
```

Purpose: starts the local RAG API on port `8000`.

Expected result:

```text
Uvicorn running on http://0.0.0.0:8000
```

Keep this command running while testing the API.

## 12. Call The RAG Endpoint

In another terminal:

```bash
poetry run poe local-rag-test
```

Purpose: sends a sample question to the local FastAPI `/rag` endpoint.

Expected result: a JSON response with an `answer` field generated by `qwen2.5:7b-instruct` using retrieved Qdrant context.

## 13. Generate A Tiny Instruction Dataset

```bash
ENV_FILE=.env.local poetry run python -m tools.local generate-dataset --dataset-type instruction --max-prompts 1
```

Purpose: tests local dataset generation through Ollama.

Expected result: a JSON file is saved to:

```text
data/generated/local_instruction_dataset.json
```

The file should contain a train/test structure with at least one instruction-answer sample.

## 14. Generate A Tiny Preference Dataset

```bash
ENV_FILE=.env.local poetry run python -m tools.local generate-dataset --dataset-type preference --max-prompts 1
```

Purpose: tests local preference dataset generation through Ollama.

Expected result: a JSON file is saved to:

```text
data/generated/local_preference_dataset.json
```

The file should contain preference samples with:

- `instruction`
- `rejected`
- `chosen`

Some generated preference samples may be filtered out by validation. That is expected.

## 15. Run Local RAG Evaluation

```bash
ENV_FILE=.env.local poetry run python -m tools.local evaluate-rag
```

Purpose: evaluates local RAG answers with simple local metrics.

Expected result: a JSON evaluation report is saved under:

```text
data/evaluations/
```

Metrics are also logged locally to MLflow under:

```text
data/mlruns/
```

## 16. Optional: Use Local LLM As Judge

```bash
ENV_FILE=.env.local poetry run python -m tools.local evaluate-rag --judge
```

Purpose: asks the local LLM to produce JSON scores for each RAG answer.

Expected result: the evaluation JSON includes a `local_judge` field with scores for:

- groundedness
- relevance
- clarity

Because this uses the local LLM as a judge, it can be slower and occasionally need retries.

## 17. Optional: View MLflow Runs

```bash
mlflow ui --backend-store-uri data/mlruns
```

Purpose: opens a local MLflow UI for evaluation runs.

Expected result: MLflow starts a local web UI, usually at:

```text
http://127.0.0.1:5000
```

## 18. Check Training Readiness

```bash
ENV_FILE=.env.local poetry run python -m tools.local training-check
```

Purpose: checks whether local fine-tuning dependencies are installed and prints the hardware profile.

Expected current result:

- `trl` missing
- `peft` missing
- `bitsandbytes` missing
- `unsloth` missing

This is expected for now. The current milestone is local RAG, dataset generation, and evaluation. Training is a later phase.

## 19. Stop Local Databases

```bash
poetry run poe local-stack-down
```

Purpose: stops MongoDB and Qdrant containers.

Expected result: Docker stops the local database containers.

Ollama is separate. Stop it by interrupting `ollama serve` with `Ctrl+C`, or leave it running if you use it elsewhere.

## Known Good Sequence

For a fresh local run, use this order:

```bash
cd /home/fspinto/projects/LLM-Engineers-Handbook
ollama serve
```

Then in another terminal:

```bash
cd /home/fspinto/projects/LLM-Engineers-Handbook
ollama list
poetry run poe local-stack-up
poetry run poe local-validate
ENV_FILE=.env.local poetry run python -m tools.local healthcheck
ENV_FILE=.env.local poetry run python -m tools.local import-data
ENV_FILE=.env.local poetry run python -m tools.local build-vector-db --reset
poetry run poe local-smoke-test
poetry run poe local-smoke-test-full
poetry run poe local-run-api
```

Then in a third terminal:

```bash
cd /home/fspinto/projects/LLM-Engineers-Handbook
poetry run poe local-rag-test
```

If all of that works, the local-first experiment is running end to end.
