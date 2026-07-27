import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import click
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from qdrant_client import QdrantClient

from llm_engineering.settings import settings


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run_command(command: list[str]) -> tuple[bool, str]:
    if shutil.which(command[0]) is None:
        return False, f"{command[0]} is not installed or not on PATH"

    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False, f"{command[0]} timed out"

    output = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, output


def _get_total_ram_gib() -> float | None:
    try:
        with Path("/proc/meminfo").open() as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return round(kib / 1024 / 1024, 1)
    except OSError:
        return None

    return None


def _check_torch_cuda() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("PyTorch CUDA", False, "torch is not importable")

    try:
        if not torch.cuda.is_available():
            return CheckResult("PyTorch CUDA", False, "CUDA is not visible to this Python process")

        device_name = torch.cuda.get_device_name(0)
        return CheckResult("PyTorch CUDA", True, f"{torch.cuda.device_count()} device(s): {device_name}")
    except Exception as exc:
        return CheckResult("PyTorch CUDA", False, str(exc))


def check_hardware() -> list[CheckResult]:
    results = []

    ram_gib = _get_total_ram_gib()
    results.append(
        CheckResult("Host", True, f"{socket.gethostname()}; CPUs: {os.cpu_count()}; RAM: {ram_gib or 'unknown'} GiB")
    )

    ok, output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    results.append(CheckResult("NVIDIA driver", ok, output or "No NVIDIA driver output"))

    ok, output = _run_command(["lspci"])
    gpu_lines = [line for line in output.splitlines() if "NVIDIA" in line or "VGA" in line or "3D" in line]
    results.append(CheckResult("PCI GPU", bool(gpu_lines), "\n".join(gpu_lines) or "No GPU found in lspci"))

    results.append(_check_torch_cuda())

    return results


def _recommend_model_profile(results: list[CheckResult]) -> str:
    nvidia = next((result for result in results if result.name == "NVIDIA driver"), None)
    torch_cuda = next((result for result in results if result.name == "PyTorch CUDA"), None)

    if nvidia and nvidia.ok and "12288" in nvidia.detail:
        return "RTX 3060 12GB visible: use a 7B/8B Q4_K_M model, 4096 context, 1 parallel request."
    if torch_cuda and torch_cuda.ok:
        return "CUDA is visible: start with a 7B Q4 model and increase context only after testing VRAM."

    return "CUDA is not visible here: use Ollama/llama.cpp if it can see the GPU, otherwise use 3B/Q4 or CPU-safe settings."


def _check_mongo() -> CheckResult:
    try:
        client = MongoClient(settings.DATABASE_HOST, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        return CheckResult("MongoDB", True, settings.DATABASE_HOST)
    except PyMongoError as exc:
        return CheckResult("MongoDB", False, str(exc))


def _check_qdrant() -> CheckResult:
    try:
        client = QdrantClient(host=settings.QDRANT_DATABASE_HOST, port=settings.QDRANT_DATABASE_PORT, timeout=2)
        client.get_collections()
        return CheckResult("Qdrant", True, f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}")
    except Exception as exc:
        return CheckResult("Qdrant", False, str(exc))


def _check_ollama() -> CheckResult:
    url = settings.OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=2) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return CheckResult("Ollama", False, f"{url}: {exc}")

    models = [model.get("name", "unknown") for model in payload.get("models", [])]
    if not models:
        return CheckResult("Ollama", True, f"{url}; running, but no local models found")

    return CheckResult("Ollama", True, f"{url}; models: {', '.join(models)}")


def check_services() -> list[CheckResult]:
    return [_check_mongo(), _check_qdrant(), _check_ollama()]


def _print_results(results: list[CheckResult]) -> None:
    for result in results:
        icon = "OK" if result.ok else "FAIL"
        click.echo(f"[{icon}] {result.name}: {result.detail}")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value

    return value


def _flatten_dataset_samples(payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    split_payload = payload.get(split, {})
    if not isinstance(split_payload, dict):
        return []

    samples = []
    for dataset_payload in split_payload.values():
        if not isinstance(dataset_payload, dict):
            continue
        dataset_samples = dataset_payload.get("samples", [])
        if isinstance(dataset_samples, list):
            samples.extend(sample for sample in dataset_samples if isinstance(sample, dict))

    return samples


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""))


def _load_prompts_from_jsonl(path: Path, max_prompts: int | None = None) -> list[str]:
    prompts = []
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("instruction") or record.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
            if max_prompts is not None and len(prompts) >= max_prompts:
                break

    return prompts


def _load_sft_samples_from_jsonl(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            instruction = record.get("instruction")
            output = record.get("output") or record.get("answer")
            if not isinstance(instruction, str) or not isinstance(output, str):
                logger.warning(f"Skipping {path}:{line_number}; expected instruction/output strings.")
                continue
            instruction = instruction.strip()
            output = output.strip()
            if not instruction or not output:
                logger.warning(f"Skipping {path}:{line_number}; instruction/output cannot be blank.")
                continue
            sample = {"instruction": instruction, "output": output}
            for key in ("source_name", "source_path", "chunk_index"):
                if key in record:
                    sample[key] = record[key]
            samples.append(sample)

    return samples


def _split_train_eval(
    samples: list[dict[str, Any]],
    *,
    test_size: float,
    min_eval_samples: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not samples:
        return [], []
    if len(samples) == 1 or test_size <= 0:
        return samples, []
    if test_size >= 1:
        raise ValueError("test_size must be smaller than 1.")

    eval_count = max(min_eval_samples, round(len(samples) * test_size))
    eval_count = min(eval_count, len(samples) - 1)

    return samples[:-eval_count], samples[-eval_count:]


def _word_overlap(answer: str, context: str) -> float:
    answer_words = {word.strip(".,!?;:()[]{}\"'").lower() for word in answer.split()}
    context_words = {word.strip(".,!?;:()[]{}\"'").lower() for word in context.split()}
    answer_words = {word for word in answer_words if len(word) > 3}
    context_words = {word for word in context_words if len(word) > 3}

    if not answer_words:
        return 0.0

    return round(len(answer_words & context_words) / len(answer_words), 4)


def _assert_ok(results: list[CheckResult]) -> None:
    failed = [result for result in results if not result.ok]
    if failed:
        details = "\n".join(f"- {result.name}: {result.detail}" for result in failed)
        raise click.ClickException(f"One or more checks failed:\n{details}")


def _is_local_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"}


def validate_local_first() -> list[CheckResult]:
    checks = [
        CheckResult("USE_CLOUD", not settings.USE_CLOUD, f"USE_CLOUD={settings.USE_CLOUD}"),
        CheckResult(
            "USE_ZENML_SECRET_STORE",
            not settings.USE_ZENML_SECRET_STORE,
            f"USE_ZENML_SECRET_STORE={settings.USE_ZENML_SECRET_STORE}",
        ),
        CheckResult("USE_OPIK", not settings.USE_OPIK, f"USE_OPIK={settings.USE_OPIK}"),
        CheckResult(
            "USE_HUGGINGFACE_HUB",
            not settings.USE_HUGGINGFACE_HUB,
            f"USE_HUGGINGFACE_HUB={settings.USE_HUGGINGFACE_HUB}",
        ),
        CheckResult("LLM_PROVIDER", settings.LLM_PROVIDER == "ollama", f"LLM_PROVIDER={settings.LLM_PROVIDER}"),
        CheckResult(
            "OLLAMA_BASE_URL",
            _is_local_http_url(settings.OLLAMA_BASE_URL),
            f"OLLAMA_BASE_URL={settings.OLLAMA_BASE_URL}",
        ),
        CheckResult(
            "MLFLOW_TRACKING_URI",
            settings.MLFLOW_TRACKING_URI.startswith("file:"),
            f"MLFLOW_TRACKING_URI={settings.MLFLOW_TRACKING_URI}",
        ),
        CheckResult(
            "MongoDB host",
            "localhost" in settings.DATABASE_HOST or "127.0.0.1" in settings.DATABASE_HOST,
            f"DATABASE_HOST={settings.DATABASE_HOST}",
        ),
        CheckResult(
            "Qdrant host",
            settings.QDRANT_DATABASE_HOST in {"localhost", "127.0.0.1"},
            f"QDRANT_DATABASE_HOST={settings.QDRANT_DATABASE_HOST}",
        ),
    ]

    return checks


def audit_legacy_cloud_references() -> list[CheckResult]:
    source_roots = [Path("llm_engineering"), Path("pipelines"), Path("steps"), Path("tools"), Path("configs")]
    patterns = {
        "aws": "AWS/SageMaker",
        "sagemaker": "AWS/SageMaker",
        "openai": "OpenAI",
        "chatopenai": "OpenAI",
        "huggingface": "Hugging Face Hub",
        "push_to_hub": "Hugging Face Hub",
        "comet": "Comet/Opik",
        "opik": "Comet/Opik",
        "zenml": "ZenML",
    }
    buckets: dict[str, set[str]] = {label: set() for label in set(patterns.values())}

    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml", ".toml"}:
                continue
            text = path.read_text(errors="ignore").lower()
            for pattern, label in patterns.items():
                if pattern in text:
                    buckets[label].add(str(path))

    results = []
    for label in sorted(buckets):
        files = sorted(buckets[label])
        if files:
            results.append(
                CheckResult(
                    f"Legacy references: {label}",
                    True,
                    f"{len(files)} file(s). These are allowed only behind local guards or legacy cloud commands.",
                )
            )

    return results


@click.group()
def main() -> None:
    """Local-first runtime checks and utilities."""


@main.command("check-hardware")
def hardware_command() -> None:
    results = check_hardware()
    _print_results(results)
    click.echo(f"\nRecommendation: {_recommend_model_profile(results)}")


@main.command("healthcheck")
def healthcheck_command() -> None:
    results = check_services()
    _print_results(results)

    _assert_ok(results)

    logger.info("All local services are ready.")


@main.command("validate")
def validate_command() -> None:
    click.echo("Validating local-first configuration...")
    config_results = validate_local_first()
    _print_results(config_results)
    _assert_ok(config_results)

    click.echo("\nAuditing legacy cloud references...")
    audit_results = audit_legacy_cloud_references()
    if audit_results:
        _print_results(audit_results)
    else:
        click.echo("[OK] Legacy references: none found")

    click.echo("\nLocal-first validation passed.")


@main.command("import-data")
@click.option(
    "--data-dir",
    default=Path("data/data_warehouse_raw_data"),
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    help="Directory containing exported MongoDB JSON documents.",
)
@click.option("--reset", is_flag=True, help="Delete existing local MongoDB documents before importing.")
def import_data_command(data_dir: Path, reset: bool) -> None:
    from llm_engineering.domain.documents import ArticleDocument, PostDocument, RepositoryDocument, UserDocument
    from llm_engineering.infrastructure.db.mongo import connection

    database = connection.get_database(settings.DATABASE_NAME)
    document_classes = {
        "ArticleDocument": ArticleDocument,
        "PostDocument": PostDocument,
        "RepositoryDocument": RepositoryDocument,
        "UserDocument": UserDocument,
    }

    if reset:
        for document_class in document_classes.values():
            database[document_class.get_collection_name()].delete_many({})
            logger.info(f"Reset MongoDB collection '{document_class.get_collection_name()}'.")

    for file in sorted(data_dir.glob("*.json")):
        document_class = document_classes.get(file.stem)
        if document_class is None:
            logger.warning(f"Skipping {file}; no matching document class.")
            continue

        existing_count = database[document_class.get_collection_name()].count_documents({})
        if existing_count > 0 and not reset:
            logger.info(
                f"Skipping {document_class.__name__}; collection already has {existing_count} document(s). "
                "Use --reset to reimport."
            )
            continue

        documents_payload = json.loads(file.read_text())
        if not documents_payload:
            logger.info(f"No documents found in {file}.")
            continue

        documents = [document_class.from_mongo(document) for document in documents_payload]
        document_class.bulk_insert(documents)
        logger.info(f"Imported {len(documents)} document(s) into '{document_class.get_collection_name()}'.")


@main.command("build-vector-db")
@click.option(
    "--authors",
    multiple=True,
    default=("Paul Iusztin", "Maxime Labonne"),
    help="Author full name to index. Can be passed multiple times.",
)
@click.option("--reset", is_flag=True, help="Delete existing local Qdrant collections before indexing.")
def build_vector_db_command(authors: tuple[str, ...], reset: bool) -> None:
    from llm_engineering.application import utils
    from llm_engineering.application.preprocessing import ChunkingDispatcher, CleaningDispatcher, EmbeddingDispatcher
    from llm_engineering.domain.base import VectorBaseDocument
    from llm_engineering.domain.documents import ArticleDocument, PostDocument, RepositoryDocument, UserDocument
    from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk, EmbeddedPostChunk, EmbeddedRepositoryChunk
    from llm_engineering.infrastructure.db.qdrant import connection

    embedded_classes = [EmbeddedArticleChunk, EmbeddedPostChunk, EmbeddedRepositoryChunk]
    if reset:
        for embedded_class in embedded_classes:
            try:
                connection.delete_collection(collection_name=embedded_class.get_collection_name())
                logger.info(f"Deleted Qdrant collection '{embedded_class.get_collection_name()}'.")
            except Exception:
                logger.info(f"Qdrant collection '{embedded_class.get_collection_name()}' did not need deletion.")

    for embedded_class in embedded_classes:
        embedded_class.get_or_create_collection()

    raw_documents = []
    for author_full_name in authors:
        first_name, last_name = utils.split_user_full_name(author_full_name)
        user = UserDocument.find(first_name=first_name, last_name=last_name)
        if user is None:
            logger.warning(f"Author '{author_full_name}' is not present in MongoDB. Skipping.")
            continue

        raw_documents.extend(ArticleDocument.bulk_find(author_id=str(user.id)))
        raw_documents.extend(PostDocument.bulk_find(author_id=str(user.id)))
        raw_documents.extend(RepositoryDocument.bulk_find(author_id=str(user.id)))

    logger.info(f"Loaded {len(raw_documents)} raw document(s) from MongoDB.")
    if not raw_documents:
        raise click.ClickException("No local documents found. Run local-import-data first.")

    cleaned_documents = [CleaningDispatcher.dispatch(document) for document in raw_documents]
    chunks = []
    for cleaned_document in cleaned_documents:
        chunks.extend(ChunkingDispatcher.dispatch(cleaned_document))
    logger.info(f"Generated {len(chunks)} chunk(s).")

    grouped_chunks = VectorBaseDocument.group_by_class(chunks)
    embedded_chunks = []
    for chunk_group in grouped_chunks.values():
        for batch in utils.misc.batch(chunk_group, size=8):
            embedded_batch = EmbeddingDispatcher.dispatch(batch)
            embedded_chunks.extend(embedded_batch)

    grouped_embeddings = VectorBaseDocument.group_by_class(embedded_chunks)
    for document_class, documents in grouped_embeddings.items():
        for batch in utils.misc.batch(documents, size=16):
            document_class.bulk_insert(batch)
        logger.info(f"Loaded {len(documents)} vector(s) into '{document_class.get_collection_name()}'.")


@main.command("generate-dataset")
@click.option(
    "--dataset-type",
    type=click.Choice(["instruction", "preference"]),
    default="instruction",
    help="Dataset type to generate locally.",
)
@click.option(
    "--cleaned-documents-file",
    default=Path("data/artifacts/cleaned_documents.json"),
    type=click.Path(path_type=Path, exists=True, dir_okay=False, file_okay=True),
    help="JSON export containing cleaned documents.",
)
@click.option("--max-prompts", default=1, show_default=True, help="Maximum prompts per category to generate.")
@click.option(
    "--output-file",
    default=None,
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    help="Output JSON file. Defaults to data/generated/local_<dataset-type>_dataset.json.",
)
def generate_dataset_command(
    dataset_type: str,
    cleaned_documents_file: Path,
    max_prompts: int,
    output_file: Path | None,
) -> None:
    from llm_engineering.application.dataset import generation
    from llm_engineering.domain.cleaned_documents import CleanedArticleDocument
    from llm_engineering.domain.dataset import DatasetType

    dataset_type_model = DatasetType(dataset_type)
    output_file = output_file or Path("data/generated") / f"local_{dataset_type}_dataset.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    exported_payload = json.loads(cleaned_documents_file.read_text())
    documents_payload = exported_payload.get("artifact_data", exported_payload)
    documents = [CleanedArticleDocument(**document) for document in documents_payload if "link" in document]

    dataset_generator = generation.get_dataset_generator(dataset_type_model)
    prompts_by_category = dataset_generator.get_prompts(documents)
    prompts_by_category = {
        category: prompts[:max_prompts] for category, prompts in prompts_by_category.items() if prompts[:max_prompts]
    }

    logger.info(
        f"Generating local {dataset_type} dataset using {sum(len(prompts) for prompts in prompts_by_category.values())} prompt(s)."
    )
    dataset = dataset_generator.generate(prompts=prompts_by_category, test_size=0.1, mock=False)
    output_file.write_text(json.dumps(_json_safe(dataset), indent=2))
    logger.info(f"Saved local dataset to {output_file}.")


@main.command("evaluate-rag")
@click.option(
    "--query",
    "queries",
    multiple=True,
    default=(
        "Draft a short post explaining how RAG works with vector databases.",
        "Explain the difference between RAG and fine tuning.",
    ),
    help="Question to evaluate. Can be passed multiple times.",
)
@click.option("--judge", is_flag=True, help="Ask the local LLM to score each answer as JSON.")
@click.option(
    "--output-file",
    default=None,
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    help="Output JSON file. Defaults to data/evaluations/local_rag_<timestamp>.json.",
)
def evaluate_rag_command(queries: tuple[str, ...], judge: bool, output_file: Path | None) -> None:
    from llm_engineering.application.llm import get_llm_provider
    from llm_engineering.application.rag.retriever import ContextRetriever
    from llm_engineering.application.utils import misc
    from llm_engineering.domain.embedded_chunks import EmbeddedChunk
    from llm_engineering.model.inference import InferenceExecutor

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_file or Path("data/evaluations") / f"local_rag_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    provider = get_llm_provider()
    retriever = ContextRetriever(mock=False)
    results = []

    for query in queries:
        documents = retriever.search(query, k=3)
        context = EmbeddedChunk.to_context(documents)
        answer = provider.generate(
            InferenceExecutor.build_prompt(query=query, context=context),
            temperature=settings.TEMPERATURE_INFERENCE,
            top_p=settings.TOP_P_INFERENCE,
            max_new_tokens=settings.MAX_NEW_TOKENS_INFERENCE,
        )
        result = {
            "query": query,
            "answer": answer,
            "retrieved_count": len(documents),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
            "context_overlap": _word_overlap(answer, context),
            "sources": [
                {
                    "type": document.__class__.__name__,
                    "platform": document.platform,
                    "author": document.author_full_name,
                    "link": getattr(document, "link", None),
                }
                for document in documents
            ],
        }

        if judge:
            judge_prompt = f"""
Evaluate the answer for the query using only the retrieved context.
Return JSON with integer scores from 1 to 5 for groundedness, relevance, and clarity, plus a short rationale.

Query:
{query}

Retrieved context:
{context}

Answer:
{answer}

Schema:
{{"groundedness": 0, "relevance": 0, "clarity": 0, "rationale": "..."}}
"""
            result["local_judge"] = provider.generate_json(
                judge_prompt,
                temperature=0,
                max_new_tokens=512,
                retries=settings.LOCAL_LLM_MAX_RETRIES,
            )

        results.append(result)

    aggregate = {
        "num_queries": len(results),
        "avg_retrieved_count": round(sum(item["retrieved_count"] for item in results) / len(results), 4),
        "avg_context_overlap": round(sum(item["context_overlap"] for item in results) / len(results), 4),
        "avg_answer_tokens": round(sum(item["answer_tokens"] for item in results) / len(results), 4),
    }
    payload = {
        "model": settings.LOCAL_CHAT_MODEL,
        "embedding_model": settings.TEXT_EMBEDDING_MODEL_ID,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "aggregate": aggregate,
        "results": results,
    }
    output_file.write_text(json.dumps(_json_safe(payload), indent=2))
    logger.info(f"Saved local RAG evaluation to {output_file}.")

    if settings.USE_MLFLOW:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"local-rag-{timestamp}"):
                mlflow.log_params(
                    {
                        "model": settings.LOCAL_CHAT_MODEL,
                        "embedding_model": settings.TEXT_EMBEDDING_MODEL_ID,
                        "num_queries": len(results),
                    }
                )
                mlflow.log_metrics(aggregate)
                mlflow.log_artifact(str(output_file))
            logger.info("Logged local RAG evaluation to MLflow.")
        except Exception:
            logger.exception("Failed to log local RAG evaluation to MLflow.")


@main.command("smoke-test")
@click.option(
    "--full",
    is_flag=True,
    help="Also run tiny local dataset generation and local RAG evaluation.",
)
def smoke_test_command(full: bool) -> None:
    from fastapi.testclient import TestClient

    from llm_engineering.application.llm import get_llm_provider
    from llm_engineering.application.rag.retriever import ContextRetriever
    from llm_engineering.infrastructure.inference_pipeline_api import app

    click.echo("Checking local services...")
    service_results = check_services()
    _print_results(service_results)
    _assert_ok(service_results)

    click.echo("\nChecking local LLM...")
    llm_answer = get_llm_provider().generate(
        "Reply with exactly: smoke ok",
        temperature=0,
        max_new_tokens=16,
    )
    if "smoke ok" not in llm_answer.lower():
        raise click.ClickException(f"Unexpected local LLM response: {llm_answer}")
    click.echo(f"[OK] Local LLM: {settings.LOCAL_CHAT_MODEL}")

    click.echo("\nChecking retrieval...")
    documents = ContextRetriever(mock=False).search("Explain RAG with vector databases.", k=3)
    if not documents:
        raise click.ClickException("RAG retrieval returned no documents. Build the vector DB first.")
    click.echo(f"[OK] Retrieval: {len(documents)} document(s)")

    click.echo("\nChecking FastAPI RAG endpoint...")
    client = TestClient(app)
    response = client.post("/rag", json={"query": "Draft a short post about RAG and vector databases."})
    if response.status_code != 200:
        raise click.ClickException(f"RAG endpoint failed with HTTP {response.status_code}: {response.text}")
    answer = response.json().get("answer", "")
    if not answer:
        raise click.ClickException("RAG endpoint returned an empty answer.")
    click.echo(f"[OK] FastAPI /rag: {answer[:120].replace(chr(10), ' ')}...")

    if full:
        click.echo("\nRunning full local checks...")
        generate_dataset_command.callback(
            dataset_type="instruction",
            cleaned_documents_file=Path("data/artifacts/cleaned_documents.json"),
            max_prompts=1,
            output_file=Path("data/generated/smoke_instruction_dataset.json"),
        )
        evaluate_rag_command.callback(
            queries=("Explain the difference between RAG and fine tuning.",),
            judge=False,
            output_file=Path("data/evaluations/smoke_rag_evaluation.json"),
        )
        click.echo("[OK] Full checks: dataset generation and evaluation")

    click.echo("\nLocal-first smoke test passed.")


@main.command("training-check")
def training_check_command() -> None:
    required_packages = ["trl", "peft", "bitsandbytes", "unsloth"]
    for package in required_packages:
        try:
            __import__(package)
            click.echo(f"[OK] {package}: installed")
        except Exception as exc:
            click.echo(f"[FAIL] {package}: {type(exc).__name__}: {exc}")

    click.echo("")
    _print_results(check_hardware())
    click.echo(
        "\nRecommended RTX 3060 12GB training profile: QLoRA, sequence length 1024 first, "
        "batch size 1, gradient accumulation 8-16, LoRA rank 8 or 16."
    )


@main.command("prepare-training-data")
@click.option(
    "--instruction-file",
    default=Path("data/generated/local_instruction_dataset.json"),
    type=click.Path(path_type=Path, exists=True, dir_okay=False, file_okay=True),
    help="Local instruction dataset JSON generated by tools.local generate-dataset.",
)
@click.option(
    "--preference-file",
    default=Path("data/generated/local_preference_dataset.json"),
    type=click.Path(path_type=Path, exists=True, dir_okay=False, file_okay=True),
    help="Local preference dataset JSON generated by tools.local generate-dataset.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="Directory for JSONL training files. Defaults to LOCAL_TRAINING_DATA_DIR.",
)
def prepare_training_data_command(instruction_file: Path, preference_file: Path, output_dir: Path | None) -> None:
    output_dir = output_dir or Path(settings.LOCAL_TRAINING_DATA_DIR)

    instruction_payload = json.loads(instruction_file.read_text())
    preference_payload = json.loads(preference_file.read_text())

    instruction_train = [
        {"instruction": sample["instruction"], "output": sample["answer"]}
        for sample in _flatten_dataset_samples(instruction_payload, "train")
        if sample.get("instruction") and sample.get("answer")
    ]
    instruction_test = [
        {"instruction": sample["instruction"], "output": sample["answer"]}
        for sample in _flatten_dataset_samples(instruction_payload, "test")
        if sample.get("instruction") and sample.get("answer")
    ]
    preference_train = [
        {
            "prompt": sample["instruction"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
        for sample in _flatten_dataset_samples(preference_payload, "train")
        if sample.get("instruction") and sample.get("chosen") and sample.get("rejected")
    ]
    preference_test = [
        {
            "prompt": sample["instruction"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
        for sample in _flatten_dataset_samples(preference_payload, "test")
        if sample.get("instruction") and sample.get("chosen") and sample.get("rejected")
    ]

    outputs = {
        "sft_train": output_dir / "sft_train.jsonl",
        "sft_test": output_dir / "sft_test.jsonl",
        "dpo_train": output_dir / "dpo_train.jsonl",
        "dpo_test": output_dir / "dpo_test.jsonl",
    }
    _write_jsonl(outputs["sft_train"], instruction_train)
    _write_jsonl(outputs["sft_test"], instruction_test)
    _write_jsonl(outputs["dpo_train"], preference_train)
    _write_jsonl(outputs["dpo_test"], preference_test)

    click.echo(f"[OK] SFT train: {len(instruction_train)} sample(s) -> {outputs['sft_train']}")
    click.echo(f"[OK] SFT test: {len(instruction_test)} sample(s) -> {outputs['sft_test']}")
    click.echo(f"[OK] DPO train: {len(preference_train)} sample(s) -> {outputs['dpo_train']}")
    click.echo(f"[OK] DPO test: {len(preference_test)} sample(s) -> {outputs['dpo_test']}")


@main.command("prepare-thesis-training-data")
@click.option(
    "--input-file",
    default=Path("data/generated/local_thesis_sft_dataset.jsonl"),
    type=click.Path(path_type=Path, exists=True, dir_okay=False, file_okay=True),
    help="Thesis/domain SFT JSONL generated by tools.local generate-thesis-dataset.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="Directory for SFT JSONL training files. Defaults to LOCAL_TRAINING_DATA_DIR.",
)
@click.option("--test-size", default=0.1, show_default=True, help="Fraction of samples reserved for eval.")
def prepare_thesis_training_data_command(input_file: Path, output_dir: Path | None, test_size: float) -> None:
    if test_size < 0 or test_size >= 1:
        raise click.ClickException("--test-size must be greater than or equal to 0 and smaller than 1.")

    output_dir = output_dir or Path(settings.LOCAL_TRAINING_DATA_DIR)
    samples = _load_sft_samples_from_jsonl(input_file)
    if not samples:
        raise click.ClickException(f"No valid thesis/domain SFT samples found in {input_file}.")

    train_samples, eval_samples = _split_train_eval(samples, test_size=test_size)
    train_file = output_dir / "sft_train.jsonl"
    eval_file = output_dir / "sft_test.jsonl"

    _write_jsonl(train_file, train_samples)
    _write_jsonl(eval_file, eval_samples)

    click.echo(f"[OK] Thesis/domain SFT train: {len(train_samples)} sample(s) -> {train_file}")
    click.echo(f"[OK] Thesis/domain SFT test: {len(eval_samples)} sample(s) -> {eval_file}")
    click.echo("Next: inspect the JSONL files, then run `poetry run poe local-train-dry-run`.")


@main.command("training-plan")
def training_plan_command() -> None:
    click.echo("Local RTX 3060 12GB training profile:")
    click.echo(f"- base model: {settings.LOCAL_TRAINING_BASE_MODEL}")
    click.echo(f"- output dir: {settings.LOCAL_TRAINING_OUTPUT_DIR}")
    click.echo(f"- data dir: {settings.LOCAL_TRAINING_DATA_DIR}")
    click.echo(f"- max sequence length: {settings.LOCAL_TRAINING_MAX_SEQ_LENGTH}")
    click.echo(f"- 4-bit loading: {settings.LOCAL_TRAINING_LOAD_IN_4BIT}")
    click.echo(
        "- LoRA rank/alpha/dropout: "
        f"{settings.LOCAL_TRAINING_LORA_RANK}/{settings.LOCAL_TRAINING_LORA_ALPHA}/"
        f"{settings.LOCAL_TRAINING_LORA_DROPOUT}"
    )
    click.echo(f"- batch size: {settings.LOCAL_TRAINING_BATCH_SIZE}")
    click.echo(f"- gradient accumulation: {settings.LOCAL_TRAINING_GRADIENT_ACCUMULATION_STEPS}")
    click.echo(f"- learning rate: {settings.LOCAL_TRAINING_LEARNING_RATE}")
    click.echo(f"- epochs: {settings.LOCAL_TRAINING_NUM_EPOCHS}")
    click.echo("\nThis is a preparation profile only. Install training dependencies before running GPU training.")


@main.command("train")
@click.option("--stage", type=click.Choice(["sft", "dpo"]), default="sft", show_default=True)
@click.option("--dry-run", is_flag=True, help="Check local training readiness without loading the model.")
def train_command(stage: str, dry_run: bool) -> None:
    from llm_engineering.model.finetuning.local import (
        LocalTrainingConfig,
        check_local_training_readiness,
        run_local_training,
    )

    config = LocalTrainingConfig.from_settings(stage=stage)
    readiness = check_local_training_readiness(config)

    click.echo(f"Local training stage: {config.stage}")
    click.echo(f"Base model: {config.base_model}")
    click.echo(f"Train file: {config.train_file}")
    click.echo(f"Eval file: {config.eval_file}")
    click.echo(f"Output dir: {config.output_dir}")

    click.echo("\nReadiness checks:")
    for check in readiness.checks:
        click.echo(f"[OK] {check}")
    for failure in readiness.failures:
        click.echo(f"[FAIL] {failure}")

    if dry_run:
        if readiness.ok:
            click.echo("\nLocal training dry run passed.")
        else:
            raise click.ClickException("Local training dry run failed. Fix the failed checks before training.")
        return

    if not readiness.ok:
        raise click.ClickException("Local training is not ready. Run with --dry-run to inspect failed checks.")

    run_local_training(config)


@main.command("infer")
@click.option(
    "--prompt",
    default=None,
    help="Instruction prompt to send to the local fine-tuned model.",
)
@click.option("--base-only", is_flag=True, help="Use the base model without the local LoRA adapter.")
@click.option(
    "--adapter-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="LoRA adapter directory. Defaults to LOCAL_TRAINING_OUTPUT_DIR/adapter.",
)
@click.option("--max-new-tokens", default=128, show_default=True, help="Maximum generated tokens.")
@click.option("--temperature", default=0.2, show_default=True, help="Sampling temperature. Use 0 for greedy output.")
def infer_command(
    prompt: str | None,
    base_only: bool,
    adapter_dir: Path | None,
    max_new_tokens: int,
    temperature: float,
) -> None:
    from llm_engineering.model.finetuning.local import (
        DEFAULT_INFERENCE_PROMPT,
        LocalTrainingConfig,
        _default_adapter_dir,
        generate_local_response,
    )

    config = LocalTrainingConfig.from_settings(stage="sft")
    prompt = prompt or DEFAULT_INFERENCE_PROMPT
    adapter_dir = None if base_only else adapter_dir or _default_adapter_dir(config)

    click.echo(f"Base model: {config.base_model}")
    if adapter_dir is None:
        click.echo("Adapter: disabled")
    else:
        click.echo(f"Adapter: {adapter_dir}")
    click.echo(f"Prompt: {prompt}\n")

    response = generate_local_response(
        prompt=prompt,
        config=config,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    click.echo(response)


@main.command("compare-adapter")
@click.option(
    "--prompt",
    "prompts",
    multiple=True,
    default=(
        "Explain what supervised fine tuning is.",
        "Explain LoRA in one short paragraph.",
    ),
    help="Prompt to compare. Can be passed multiple times.",
)
@click.option(
    "--prompts-file",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False, file_okay=True),
    help="JSONL file with 'instruction' or 'prompt' fields to compare.",
)
@click.option("--max-prompts", default=3, show_default=True, help="Maximum prompts to read from --prompts-file.")
@click.option(
    "--adapter-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="LoRA adapter directory. Defaults to LOCAL_TRAINING_OUTPUT_DIR/adapter.",
)
@click.option("--max-new-tokens", default=96, show_default=True, help="Maximum generated tokens.")
@click.option("--temperature", default=0.0, show_default=True, help="Sampling temperature. Use 0 for greedy output.")
@click.option(
    "--output-file",
    default=None,
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    help="Output JSON report. Defaults to data/training/evaluations/local_adapter_comparison_<timestamp>.json.",
)
def compare_adapter_command(
    prompts: tuple[str, ...],
    prompts_file: Path | None,
    max_prompts: int,
    adapter_dir: Path | None,
    max_new_tokens: int,
    temperature: float,
    output_file: Path | None,
) -> None:
    from llm_engineering.model.finetuning.local import (
        LocalTrainingConfig,
        _default_adapter_dir,
        generate_local_response,
    )

    config = LocalTrainingConfig.from_settings(stage="sft")
    adapter_dir = adapter_dir or _default_adapter_dir(config)
    if prompts_file is not None:
        prompts = tuple(_load_prompts_from_jsonl(prompts_file, max_prompts=max_prompts))
        if not prompts:
            raise click.ClickException(f"No prompts found in {prompts_file}.")

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_file or Path("data/training/evaluations") / f"local_adapter_comparison_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for prompt in prompts:
        click.echo(f"Comparing prompt: {prompt}")
        base_response = generate_local_response(
            prompt=prompt,
            config=config,
            adapter_dir=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        adapter_response = generate_local_response(
            prompt=prompt,
            config=config,
            adapter_dir=adapter_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results.append(
            {
                "prompt": prompt,
                "base_response": base_response,
                "adapter_response": adapter_response,
                "identical": base_response == adapter_response,
            }
        )

    payload = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "base_model": config.base_model,
        "adapter_dir": str(adapter_dir),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "results": results,
    }
    output_file.write_text(json.dumps(payload, indent=2))

    click.echo(f"\nSaved comparison report to {output_file}")
    identical_count = sum(result["identical"] for result in results)
    click.echo(f"Identical responses: {identical_count}/{len(results)}")


@main.command("import-sources")
@click.option(
    "--source-dir",
    default=Path("data/local_sources"),
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="Directory containing local thesis/source files.",
)
@click.option("--reset", is_flag=True, help="Reset the local source Qdrant collection before importing.")
@click.option("--chunk-size", default=1200, show_default=True, help="Approximate chunk size in characters.")
@click.option("--overlap", default=180, show_default=True, help="Chunk overlap in characters.")
def import_sources_command(source_dir: Path, reset: bool, chunk_size: int, overlap: int) -> None:
    from llm_engineering.application.local_sources import (
        build_local_source_chunks,
        load_local_source_documents,
        reset_local_source_collection,
        save_source_manifest,
        upsert_local_source_chunks,
    )

    documents = load_local_source_documents(source_dir)
    if not documents:
        click.echo(f"No supported local sources found in {source_dir}. Add .pdf, .txt, or .md files and rerun.")
        return

    if reset:
        reset_local_source_collection()

    chunks = build_local_source_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    upsert_local_source_chunks(chunks)
    save_source_manifest(
        source_dir=source_dir,
        output_file=Path("data/local_sources_manifest.json"),
        documents=documents,
    )

    click.echo(f"[OK] Documents: {len(documents)}")
    click.echo(f"[OK] Chunks: {len(chunks)}")
    click.echo("[OK] Qdrant collection: local_sources")


@main.command("search-sources")
@click.option("--query", default="What is the thesis about?", show_default=True)
@click.option("--limit", default=5, show_default=True)
def search_sources_command(query: str, limit: int) -> None:
    from llm_engineering.application.local_sources import search_local_sources

    results = search_local_sources(query=query, limit=limit)
    if not results:
        raise click.ClickException("No local source results found. Run local-import-sources first.")

    for index, result in enumerate(results, start=1):
        preview = result["content"][:500].replace("\n", " ")
        click.echo(f"\n[{index}] {result['source_name']} chunk={result['chunk_index']} score={result['score']:.4f}")
        click.echo(preview)


@main.command("ask-sources")
@click.option("--question", prompt=True, help="Question to answer from the local thesis/source corpus.")
@click.option("--limit", default=5, show_default=True, help="Number of local source chunks to retrieve.")
@click.option("--temperature", default=0.1, show_default=True, help="Local LLM temperature.")
@click.option("--max-new-tokens", default=700, show_default=True, help="Maximum answer tokens to generate.")
@click.option(
    "--retrieval-query",
    "retrieval_queries",
    multiple=True,
    help="Extra retrieval query. Can be provided multiple times for targeted questions.",
)
def ask_sources_command(
    question: str,
    limit: int,
    temperature: float,
    max_new_tokens: int,
    retrieval_queries: tuple[str, ...],
) -> None:
    from llm_engineering.application.llm import get_llm_provider
    from llm_engineering.application.local_sources import search_local_sources

    default_retrieval_queries = (
        question,
        (
            f"{question} Francisco Pinto thesis managerial adjustments workforce scheduling "
            "optimizer-generated human-approved schedules research questions methodology results"
        ),
    )
    queries = (*default_retrieval_queries, *retrieval_queries)
    candidate_results = []
    for query in queries:
        candidate_results.extend(search_local_sources(query=query, limit=limit))

    results = dedupe_chunks(sorted(candidate_results, key=lambda result: result["score"], reverse=True))[:limit]
    if not results:
        raise click.ClickException("No local source results found. Run local-import-sources first.")

    context = "\n\n".join(
        f"[{index}] Source: {result['source_name']} | chunk={result['chunk_index']}\n{result['content'][:1800]}"
        for index, result in enumerate(results, start=1)
    )
    prompt = f"""
You are answering questions about Francisco Pinto's thesis and its supporting literature.
Answer using only the context below. If the context is not enough, say what is missing.
Prefer the thesis as the primary source when it appears in the context.
Be concise, precise, and avoid inventing citations or results.

Question:
{question}

Context:
{context}
"""
    answer = get_llm_provider().generate(
        prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    click.echo("\nAnswer:\n")
    click.echo(answer)
    click.echo("\nSources:")
    for index, result in enumerate(results, start=1):
        click.echo(f"- [{index}] {result['source_name']} chunk={result['chunk_index']} score={result['score']:.4f}")


@main.command("generate-thesis-dataset")
@click.option(
    "--output-file",
    default=Path("data/generated/local_thesis_sft_dataset.jsonl"),
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    help="Output JSONL file for local thesis/domain SFT samples.",
)
@click.option("--max-chunks", default=10, show_default=True, help="Maximum local source chunks to sample.")
@click.option("--questions-per-chunk", default=2, show_default=True, help="Questions to generate per chunk.")
@click.option(
    "--thesis-source-name",
    default="Predicting_Managerial_Adjustments_Francisco_Pinto.pdf",
    show_default=True,
    help="Primary thesis PDF filename used for weighted generation.",
)
@click.option("--thesis-ratio", default=0.7, show_default=True, help="Approximate share of chunks from the thesis.")
@click.option(
    "--literature-ratio",
    default=0.2,
    show_default=True,
    help="Approximate share of chunks from supporting literature.",
)
@click.option(
    "--seed-query",
    "seed_queries",
    multiple=True,
    help="Retrieval query used to select source chunks. Can be provided multiple times.",
)
def generate_thesis_dataset_command(
    output_file: Path,
    max_chunks: int,
    questions_per_chunk: int,
    thesis_source_name: str,
    thesis_ratio: float,
    literature_ratio: float,
    seed_queries: tuple[str, ...],
) -> None:
    from llm_engineering.application.llm import get_llm_provider
    from llm_engineering.application.local_sources import search_local_sources

    provider = get_llm_provider()
    if not seed_queries:
        seed_queries = (
            "managerial adjustments demand forecasting labor scheduling methodology research questions",
            "empirical characterization managerial adjustments solution gap target variables PROC10",
            "machine learning model predicts managerial adjustments features evaluation results",
            "literature review human algorithm interaction labor scheduling managerial overrides",
        )
    thesis_chunk_count = max(1, round(max_chunks * thesis_ratio))
    literature_chunk_count = max(0, round(max_chunks * literature_ratio))
    synthesis_chunk_count = max(0, max_chunks - thesis_chunk_count - literature_chunk_count)

    if thesis_chunk_count > max_chunks:
        thesis_chunk_count = max_chunks
        literature_chunk_count = 0
        synthesis_chunk_count = 0

    thesis_chunks = []
    literature_chunks = []
    synthesis_chunks = []
    for query in seed_queries:
        thesis_chunks.extend(search_local_sources(query=query, limit=max_chunks, source_name=thesis_source_name))
        literature_chunks.extend(
            search_local_sources(query=query, limit=max_chunks, exclude_source_name=thesis_source_name)
        )
        synthesis_chunks.extend(search_local_sources(query=query, limit=max_chunks))

    selected_chunks = [
        *dedupe_chunks(thesis_chunks)[:thesis_chunk_count],
        *dedupe_chunks(literature_chunks)[:literature_chunk_count],
        *dedupe_chunks(synthesis_chunks)[:synthesis_chunk_count],
    ]
    selected_chunks = dedupe_chunks(selected_chunks)[:max_chunks]
    if len(selected_chunks) < max_chunks:
        fill_candidates = dedupe_chunks([*thesis_chunks, *literature_chunks, *synthesis_chunks])
        selected_keys = {(chunk["source_path"], chunk["chunk_index"]) for chunk in selected_chunks}
        for chunk in fill_candidates:
            key = (chunk["source_path"], chunk["chunk_index"])
            if key in selected_keys:
                continue
            selected_chunks.append(chunk)
            selected_keys.add(key)
            if len(selected_chunks) == max_chunks:
                break

    if not selected_chunks:
        raise click.ClickException("No local source chunks found. Run local-import-sources first.")

    thesis_selected = sum(1 for chunk in selected_chunks if chunk.get("source_name") == thesis_source_name)
    literature_selected = sum(1 for chunk in selected_chunks if chunk.get("source_name") != thesis_source_name)
    click.echo(
        "[OK] Selected chunks: "
        f"{len(selected_chunks)} total, {thesis_selected} thesis, {literature_selected} literature/synthesis"
    )

    samples = []
    for chunk in selected_chunks:
        prompt = f"""
Create {questions_per_chunk} supervised fine-tuning samples from the context.
The model should become better at explaining the thesis/domain.
Return only JSON as a list. Each item must have "instruction" and "output".
The output must be grounded in the context and should not mention that it was generated from a chunk.

Source: {chunk["source_name"]}
Context:
{chunk["content"][:3000]}
"""
        generated = provider.generate_json(
            prompt,
            temperature=0.2,
            max_new_tokens=1200,
            retries=settings.LOCAL_LLM_MAX_RETRIES,
        )
        if not isinstance(generated, list):
            logger.warning("Local LLM returned non-list JSON. Skipping chunk.")
            continue
        for item in generated:
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction")
            output = item.get("output") or item.get("answer")
            if isinstance(instruction, str) and isinstance(output, str) and instruction.strip() and output.strip():
                samples.append(
                    {
                        "instruction": instruction.strip(),
                        "output": output.strip(),
                        "source_name": chunk["source_name"],
                        "source_path": chunk["source_path"],
                        "chunk_index": chunk["chunk_index"],
                    }
                )

    if not samples:
        raise click.ClickException("No thesis SFT samples were generated.")

    _write_jsonl(output_file, samples)
    click.echo(f"[OK] Generated {len(samples)} thesis/domain SFT sample(s) -> {output_file}")


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    deduped_chunks = {}
    for chunk in chunks:
        key = (chunk["source_path"], chunk["chunk_index"])
        deduped_chunks[key] = chunk

    return list(deduped_chunks.values())


if __name__ == "__main__":
    main()
