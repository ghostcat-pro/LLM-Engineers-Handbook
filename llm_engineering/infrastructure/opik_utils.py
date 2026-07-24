import os
from collections.abc import Callable
from typing import Any

import opik
from loguru import logger
from opik.configurator.configure import OpikConfigurator

from llm_engineering import settings


def track(func: Callable | None = None, **kwargs: Any) -> Callable:
    if settings.USE_OPIK:
        return opik.track(func, **kwargs) if func is not None else opik.track(**kwargs)

    def decorator(inner_func: Callable) -> Callable:
        return inner_func

    return decorator(func) if func is not None else decorator


def configure_opik() -> None:
    if not settings.USE_OPIK:
        logger.info("Opik is disabled.")
        return

    if settings.COMET_API_KEY and settings.COMET_PROJECT:
        try:
            client = OpikConfigurator(api_key=settings.COMET_API_KEY)
            default_workspace = client._get_default_workspace()
        except Exception:
            logger.warning("Default workspace not found. Setting workspace to None and enabling interactive mode.")
            default_workspace = None

        os.environ["OPIK_PROJECT_NAME"] = settings.COMET_PROJECT

        opik.configure(api_key=settings.COMET_API_KEY, workspace=default_workspace, use_local=False, force=True)
        logger.info("Opik configured successfully.")
    else:
        logger.warning(
            "COMET_API_KEY and COMET_PROJECT are not set. Set them to enable prompt monitoring with Opik (powered by Comet ML)."
        )
