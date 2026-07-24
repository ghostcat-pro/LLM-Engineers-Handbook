from fastapi import FastAPI, HTTPException
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.llm import get_llm_provider
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik, track
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

configure_opik()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@track
def call_llm_service(query: str, context: str | None) -> str:
    if not settings.USE_CLOUD:
        provider = get_llm_provider()

        return provider.generate(
            InferenceExecutor.build_prompt(query=query, context=context),
            temperature=settings.TEMPERATURE_INFERENCE,
            top_p=settings.TOP_P_INFERENCE,
            max_new_tokens=settings.MAX_NEW_TOKENS_INFERENCE,
        )

    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    return InferenceExecutor(llm, query, context).execute()


@track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    if settings.USE_OPIK:
        opik_context.update_current_trace(
            tags=["rag"],
            metadata={
                "model_id": settings.LOCAL_CHAT_MODEL if not settings.USE_CLOUD else settings.HF_MODEL_ID,
                "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "query_tokens": misc.compute_num_tokens(query),
                "context_tokens": misc.compute_num_tokens(context),
                "answer_tokens": misc.compute_num_tokens(answer),
            },
        )

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
