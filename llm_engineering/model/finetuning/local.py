import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from llm_engineering.settings import settings

REQUIRED_TRAINING_PACKAGES = ("trl", "peft", "bitsandbytes")
DEFAULT_INFERENCE_PROMPT = "Explain what supervised fine tuning is."


@dataclass(frozen=True)
class LocalTrainingConfig:
    stage: Literal["sft", "dpo"]
    base_model: str
    output_dir: Path
    data_dir: Path
    max_seq_length: int
    load_in_4bit: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int

    @classmethod
    def from_settings(cls, stage: Literal["sft", "dpo"] = "sft") -> "LocalTrainingConfig":
        return cls(
            stage=stage,
            base_model=settings.LOCAL_TRAINING_BASE_MODEL,
            output_dir=Path(settings.LOCAL_TRAINING_OUTPUT_DIR),
            data_dir=Path(settings.LOCAL_TRAINING_DATA_DIR),
            max_seq_length=settings.LOCAL_TRAINING_MAX_SEQ_LENGTH,
            load_in_4bit=settings.LOCAL_TRAINING_LOAD_IN_4BIT,
            lora_rank=settings.LOCAL_TRAINING_LORA_RANK,
            lora_alpha=settings.LOCAL_TRAINING_LORA_ALPHA,
            lora_dropout=settings.LOCAL_TRAINING_LORA_DROPOUT,
            batch_size=settings.LOCAL_TRAINING_BATCH_SIZE,
            gradient_accumulation_steps=settings.LOCAL_TRAINING_GRADIENT_ACCUMULATION_STEPS,
            learning_rate=settings.LOCAL_TRAINING_LEARNING_RATE,
            num_epochs=settings.LOCAL_TRAINING_NUM_EPOCHS,
        )

    @property
    def train_file(self) -> Path:
        filename = "sft_train.jsonl" if self.stage == "sft" else "dpo_train.jsonl"
        return self.data_dir / filename

    @property
    def eval_file(self) -> Path:
        filename = "sft_test.jsonl" if self.stage == "sft" else "dpo_test.jsonl"
        return self.data_dir / filename


@dataclass(frozen=True)
class LocalTrainingReadiness:
    ok: bool
    checks: list[str]
    failures: list[str]


def _package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open() as file:
        for line in file:
            if line.strip():
                json.loads(line)
                count += 1

    return count


def _is_local_or_cached_model(model_name_or_path: str) -> bool:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        return True

    if "/" not in model_name_or_path:
        return False

    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name_or_path.replace('/', '--')}"
    return cache_path.exists()


def _format_sft_prompt(instruction: str) -> str:
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def _default_adapter_dir(config: LocalTrainingConfig) -> Path:
    return config.output_dir / "adapter"


def check_local_training_readiness(config: LocalTrainingConfig) -> LocalTrainingReadiness:
    checks = []
    failures = []

    if not settings.USE_CLOUD:
        checks.append("USE_CLOUD=false")
    else:
        failures.append("USE_CLOUD must be false for local training.")

    if settings.USE_HUGGINGFACE_HUB:
        failures.append("USE_HUGGINGFACE_HUB must be false so training cannot push to the Hub.")
    else:
        checks.append("Hugging Face Hub upload disabled")

    for package in REQUIRED_TRAINING_PACKAGES:
        if _package_available(package):
            checks.append(f"{package} installed")
        else:
            failures.append(f"{package} is not installed")

    try:
        import torch

        if torch.cuda.is_available():
            checks.append(f"CUDA visible: {torch.cuda.get_device_name(0)}")
        else:
            failures.append("CUDA is not visible to this Python process.")
    except Exception as exc:
        failures.append(f"PyTorch CUDA check failed: {exc}")

    if _is_local_or_cached_model(config.base_model):
        checks.append(f"Base model is local/cached: {config.base_model}")
    else:
        failures.append(
            f"Base model is not local/cached: {config.base_model}. "
            "Use a local model path or pre-populate the Hugging Face cache before training."
        )

    train_count = _count_jsonl(config.train_file)
    if train_count > 0:
        checks.append(f"Training data: {train_count} sample(s) in {config.train_file}")
    else:
        failures.append(f"Training data is empty or missing: {config.train_file}")

    eval_count = _count_jsonl(config.eval_file)
    checks.append(f"Evaluation data: {eval_count} sample(s) in {config.eval_file}")

    if config.max_seq_length > 1024:
        failures.append("max_seq_length is above 1024. Start at 1024 on RTX 3060 12GB.")
    else:
        checks.append(f"max_seq_length={config.max_seq_length}")

    if config.batch_size != 1:
        failures.append("batch_size should start at 1 on RTX 3060 12GB.")
    else:
        checks.append("batch_size=1")

    if config.lora_rank > 16:
        failures.append("LoRA rank should start at 8 or 16 on RTX 3060 12GB.")
    else:
        checks.append(f"LoRA rank={config.lora_rank}")

    return LocalTrainingReadiness(ok=not failures, checks=checks, failures=failures)


def run_local_sft(config: LocalTrainingConfig) -> None:
    readiness = check_local_training_readiness(config)
    if not readiness.ok:
        failures = "\n".join(f"- {failure}" for failure in readiness.failures)
        raise RuntimeError(f"Local training is not ready:\n{failures}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    data_files = {"train": str(config.train_file)}
    if _count_jsonl(config.eval_file) > 0:
        data_files["test"] = str(config.eval_file)

    dataset = load_dataset("json", data_files=data_files)

    def format_sft_sample(example: dict) -> dict:
        return {"text": f"{_format_sft_prompt(example['instruction'])}{example['output']}{tokenizer.eos_token}"}

    dataset = dataset.map(format_sft_sample, remove_columns=dataset["train"].column_names)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=10,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        fp16=True,
        optim="paged_adamw_8bit",
        seed=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        peft_config=peft_config,
        args=training_args,
    )

    logger.info("Starting local SFT training.")
    trainer.train()
    trainer.save_model(str(config.output_dir / "adapter"))
    tokenizer.save_pretrained(str(config.output_dir / "adapter"))
    logger.info(f"Saved local adapter to {config.output_dir / 'adapter'}.")


def run_local_training(config: LocalTrainingConfig) -> None:
    if config.stage == "sft":
        run_local_sft(config)
        return

    raise NotImplementedError("Local DPO training is not implemented yet. Start with local SFT.")


def generate_local_response(
    prompt: str,
    config: LocalTrainingConfig,
    *,
    adapter_dir: Path | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not _is_local_or_cached_model(config.base_model):
        raise RuntimeError(f"Base model is not local/cached: {config.base_model}")

    if adapter_dir is not None and not adapter_dir.exists():
        raise RuntimeError(f"Adapter directory does not exist: {adapter_dir}")

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer_source = adapter_dir if adapter_dir is not None else config.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
    model.eval()

    encoded = tokenizer(_format_sft_prompt(prompt), return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
