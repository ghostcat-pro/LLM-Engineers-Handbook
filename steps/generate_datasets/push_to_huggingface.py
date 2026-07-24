from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.domain.dataset import InstructTrainTestSplit, PreferenceTrainTestSplit
from llm_engineering.settings import settings


@step
def push_to_huggingface(
    dataset: Annotated[InstructTrainTestSplit | PreferenceTrainTestSplit, "dataset_split"],
    dataset_id: Annotated[str, "dataset_id"],
) -> None:
    if not settings.USE_HUGGINGFACE_HUB:
        logger.warning("Hugging Face Hub uploads are disabled. Skipping dataset push.")
        return

    assert dataset_id is not None, "Dataset id must be provided for pushing to Huggingface"
    assert (
        settings.HUGGINGFACE_ACCESS_TOKEN is not None
    ), "Huggingface access token must be provided for pushing to Huggingface"

    logger.info(f"Pushing dataset {dataset_id} to Hugging Face.")

    huggingface_dataset = dataset.to_huggingface(flatten=True)
    huggingface_dataset.push_to_hub(dataset_id, token=settings.HUGGINGFACE_ACCESS_TOKEN)
