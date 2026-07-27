# Local Thesis And Source Documents

Use this workflow to add your thesis and supporting articles without external services.

## Source Folder

Place files here:

```text
data/local_sources/
```

Supported first-pass formats:

- `.pdf`
- `.txt`
- `.md`

PDF extraction uses local `pypdf`. No Google, AWS, OpenAI, or managed document parser is used.

## Import Sources

```bash
poetry run poe local-import-sources
```

Purpose:

- extracts text from supported files
- chunks the text locally
- embeds chunks locally
- stores vectors in the Qdrant collection `local_sources`

Expected output:

```text
[OK] Documents: ...
[OK] Chunks: ...
[OK] Qdrant collection: local_sources
```

If no files exist yet, the command creates the folder and tells you to add files.

## Search Sources

```bash
poetry run poe local-search-sources
```

Custom query:

```bash
ENV_FILE=.env.local poetry run python -m tools.local search-sources --query "What is my thesis methodology?"
```

Purpose: verifies that Qdrant retrieval works over your thesis/articles before generating training data.

## Ask Questions

Interactive prompt:

```bash
poetry run poe local-ask-sources
```

One-off question:

```bash
ENV_FILE=.env.local poetry run python -m tools.local ask-sources \
  --question "What is the main objective of the thesis?"
```

Purpose: retrieves local thesis/article chunks, sends only those chunks to the local Ollama model, and prints the answer with the source chunks used.

For hard questions, add retrieval hints:

```bash
ENV_FILE=.env.local poetry run python -m tools.local ask-sources \
  --question "How does the thesis define the action space?" \
  --retrieval-query "RQ1.1 taxonomy manual adjustments retail workforce scheduling"
```

## Evaluate Thesis RAG

Run the local thesis RAG evaluation set:

```bash
poetry run poe local-evaluate-thesis-rag
```

Purpose: asks a tracked set of thesis-specific questions, retrieves local source chunks, answers with the local Ollama model, and writes a JSON report under:

```text
data/evaluations/
```

The tracked evaluation set lives at:

```text
data/evaluation/thesis_eval_questions.jsonl
```

Each case includes expected answer points. The evaluator reports expected-point coverage, context overlap, retrieved sources, and aggregate scores. Treat the scores as a practical local regression signal, not as a perfect human-quality metric.

Quick test with only a few cases:

```bash
ENV_FILE=.env.local poetry run python -m tools.local evaluate-thesis-rag --max-cases 3
```

## Generate Thesis SFT Samples

```bash
poetry run poe local-generate-thesis-dataset
```

Purpose: uses the local LLM through Ollama to create instruction/output samples from retrieved source chunks.

By default this is thesis-weighted:

- about 70% thesis chunks from `Predicting_Managerial_Adjustments_Francisco_Pinto.pdf`
- about 20% supporting literature chunks
- remaining chunks for synthesis/comparison retrieval

Focused generation:

```bash
ENV_FILE=.env.local poetry run python -m tools.local generate-thesis-dataset \
  --seed-query "managerial adjustments demand forecasting labor scheduling methodology" \
  --seed-query "machine learning model predicts managerial adjustments features evaluation results"
```

Custom weighting:

```bash
ENV_FILE=.env.local poetry run python -m tools.local generate-thesis-dataset \
  --max-chunks 12 \
  --thesis-ratio 0.75 \
  --literature-ratio 0.15
```

Default output:

```text
data/generated/local_thesis_sft_dataset.jsonl
```

This is synthetic training data. You should inspect it before mixing it into the SFT dataset.

## Prepare Thesis SFT Training Files

After inspecting the generated JSONL, convert it into the local trainer's expected split files:

```bash
poetry run poe local-prepare-thesis-training-data
```

Expected outputs:

```text
data/training/datasets/sft_train.jsonl
data/training/datasets/sft_test.jsonl
```

The split is deterministic and local. By default, 10% of valid samples are reserved for evaluation while keeping at least one training sample.

Custom split:

```bash
ENV_FILE=.env.local poetry run python -m tools.local prepare-thesis-training-data \
  --input-file data/generated/local_thesis_sft_dataset.jsonl \
  --test-size 0.2
```

## Recommended Human Work

You do not need to manually write hundreds of samples. But it is worth manually writing or reviewing:

- 20-50 key thesis questions
- expected short answers
- terms that must be defined correctly
- claims the model must not hallucinate

Use those as an evaluation set, not necessarily as training data.

## Suggested First Real Experiment

1. Add your thesis PDF and 3-10 core article PDFs to `data/local_sources/`.
2. Run `poetry run poe local-import-sources`.
3. Run `poetry run poe local-search-sources`.
4. Run `poetry run poe local-evaluate-thesis-rag` to capture the baseline RAG behavior.
5. Run `poetry run poe local-generate-thesis-dataset`.
6. Inspect `data/generated/local_thesis_sft_dataset.jsonl`.
7. Run `poetry run poe local-prepare-thesis-training-data`.
8. Run `poetry run poe local-train-dry-run`.
9. Train locally only after the dry run passes.
