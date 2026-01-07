import time
from typing import List, Type, TypeVar

from erc3 import ERC3, TaskInfo
from google import genai
from google.genai import types
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class MyLLM:
    client: genai.Client
    api: ERC3
    task: TaskInfo
    model: str
    max_tokens: int

    def __init__(self, api: ERC3, model:str, task: TaskInfo, max_tokens=40000) -> None:
        self.api = api
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self.client = genai.Client()

    def query(self, prompt: str, response_format: Type[T], model: str = None) -> T:
        started = time.time()

        resp = self.client.models.generate_content(
            model=model or self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_format,
                temperature=0,
            ),
        )

        usage = resp.usage_metadata
        self.api.log_llm(
            task_id=self.task.task_id,
            model=model or self.model,
            duration_sec=time.time() - started,
            completion=resp.text,
            prompt_tokens=usage.prompt_token_count,
            completion_tokens=usage.candidates_token_count,
            cached_prompt_tokens=usage.cached_content_token_count or 0,
        )

        return response_format.model_validate_json(resp.text)
