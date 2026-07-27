from importlib import import_module

__all__ = [
    "generate_intruction_dataset",
    "generate_preference_dataset",
    "create_prompts",
    "push_to_huggingface",
    "query_feature_store",
]

_MODULES = {
    "create_prompts": "create_prompts",
    "generate_intruction_dataset": "generate_intruction_dataset",
    "generate_preference_dataset": "generate_preference_dataset",
    "push_to_huggingface": "push_to_huggingface",
    "query_feature_store": "query_feature_store",
}


def __getattr__(name: str):
    if name in _MODULES:
        module = import_module(f"{__name__}.{_MODULES[name]}")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
