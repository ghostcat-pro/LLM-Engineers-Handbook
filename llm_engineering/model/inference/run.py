from __future__ import annotations

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings


class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            """
        else:
            self.prompt = prompt

    @staticmethod
    def build_prompt(query: str, context: str | None = None, prompt: str | None = None) -> str:
        context = context if context else ""
        prompt = (
            prompt
            or """
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            """
        )

        return prompt.format(query=query, context=context)

    def execute(self) -> str:
        self.llm.set_payload(
            inputs=self.build_prompt(query=self.query, context=self.context, prompt=self.prompt),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]

        return answer
