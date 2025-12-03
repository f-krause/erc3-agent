import time
from typing import List, Type, TypeVar

from erc3 import ERC3, TaskInfo
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class MyLLM:
    client: OpenAI
    api: ERC3
    task: TaskInfo
    model: str
    max_tokens: int

    def __init__(self, api: ERC3, model:str, task: TaskInfo, max_tokens=40000) -> None:
        self.api = api
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self.client = OpenAI()


    def query(self, messages: List, response_format: Type[T]) -> T:

        started = time.time()
        resp = self.client.beta.chat.completions.parse(messages=messages, model=self.model, response_format=response_format, max_tokens=self.max_tokens)

        self.api.log_llm(task_id=self.task.task_id, model=self.model,duration_sec=time.time() - started, usage=resp.usage)

        return resp.choices[0].message.parsed
