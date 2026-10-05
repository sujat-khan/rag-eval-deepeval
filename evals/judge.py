# evals/judge.py
import asyncio
import re
import time
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge wrapper for ChatGroq models with automatic rate-limit retry and pacing."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0,
        max_retries: int = 8,
        max_concurrent: int = 1,
        delay: float = 2.0,
    ):
        self.model_name_str = model_name
        self.chat_model = ChatGroq(model=model_name, temperature=temperature)
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.delay = delay
        self._semaphore = None
        super().__init__(model_name=model_name)

    def _get_semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    def load_model(self):
        return self.chat_model

    def _parse_retry_after(self, error: Exception) -> float:
        msg = str(error)
        match = re.search(r"try again in ([\d\.]+)s", msg)
        if match:
            try:
                return float(match.group(1)) + 1.0
            except ValueError:
                pass
        return 3.0

    def generate(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                res = self.chat_model.invoke(prompt)
                if self.delay > 0:
                    time.sleep(self.delay)
                return str(res.content)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = self._parse_retry_after(e) + (attempt * 1.5)
                    print(f"\n[GroqJudge] Rate limit (429) hit. Pausing {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise

    async def a_generate(self, prompt: str) -> str:
        sem = self._get_semaphore()
        async with sem:
            for attempt in range(self.max_retries):
                try:
                    res = await self.chat_model.ainvoke(prompt)
                    if self.delay > 0:
                        await asyncio.sleep(self.delay)
                    return str(res.content)
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        if attempt == self.max_retries - 1:
                            raise
                        wait_time = self._parse_retry_after(e) + (attempt * 1.5)
                        print(f"\n[GroqJudge] Rate limit (429) hit. Pausing {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

    def get_model_name(self) -> str:
        return self.model_name_str
