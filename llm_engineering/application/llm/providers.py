import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> str:
        pass

    def generate_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        retries: int = 2,
    ) -> dict | list:
        last_error: Exception | None = None
        json_prompt = (
            f"{prompt}\n\n"
            "Return only valid JSON. Do not include markdown fences, comments, prose, or any text outside JSON."
        )

        for _ in range(retries + 1):
            response = self.generate(
                json_prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            try:
                return extract_json(response)
            except ValueError as exc:
                last_error = exc
                json_prompt = (
                    f"{prompt}\n\n"
                    "Your previous response was not valid JSON. Return only valid JSON matching the requested schema."
                )

        raise RuntimeError("Failed to generate valid JSON with the local LLM.") from last_error


def extract_json(text: str) -> dict | list:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    candidates = []
    for start_token, end_token in [("[", "]"), ("{", "}")]:
        start = stripped.find(start_token)
        end = stripped.rfind(end_token)
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid JSON object or array found in response: {text[:500]}")


@dataclass
class OllamaLLMProvider(LLMProvider):
    base_url: str
    model: str
    timeout_seconds: int = 120

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> str:
        url = self.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
            },
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
        except (URLError, OSError) as exc:
            raise RuntimeError(f"Ollama is not reachable at {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

        generated_text = response_payload.get("response")
        if not isinstance(generated_text, str):
            raise RuntimeError(f"Ollama response did not contain text: {response_payload}")

        return generated_text.strip()
