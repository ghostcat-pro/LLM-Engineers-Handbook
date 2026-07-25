import sys

import pytest
from click.testing import CliRunner

from llm_engineering import settings


def test_legacy_runner_is_blocked_before_pipeline_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.run as run_tool

    monkeypatch.setattr(settings, "USE_CLOUD", False)
    sys.modules.pop("pipelines", None)

    result = CliRunner().invoke(run_tool.main, ["--run-training"])

    assert result.exit_code != 0
    assert "legacy ZenML pipeline runner is disabled" in result.output
    assert "pipelines" not in sys.modules


def test_call_llm_service_uses_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_engineering.infrastructure import inference_pipeline_api

    class FakeProvider:
        def generate(self, prompt: str, **kwargs: object) -> str:
            self.prompt = prompt
            self.kwargs = kwargs
            return "local response"

    def fail_sagemaker(*args: object, **kwargs: object) -> None:
        raise AssertionError("SageMaker should not be used in local mode")

    provider = FakeProvider()
    monkeypatch.setattr(settings, "USE_CLOUD", False)
    monkeypatch.setattr(inference_pipeline_api, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(inference_pipeline_api, "LLMInferenceSagemakerEndpoint", fail_sagemaker)

    answer = inference_pipeline_api.call_llm_service("What is RAG?", "local context")

    assert answer == "local response"
    assert "What is RAG?" in provider.prompt
    assert "local context" in provider.prompt


def test_push_to_huggingface_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from steps.generate_datasets.push_to_huggingface import push_to_huggingface

    class Dataset:
        def to_huggingface(self, flatten: bool = False) -> object:
            raise AssertionError("Dataset should not be converted when Hub uploads are disabled")

    monkeypatch.setattr(settings, "USE_HUGGINGFACE_HUB", False)

    push_to_huggingface.entrypoint(Dataset(), "local/dataset")
