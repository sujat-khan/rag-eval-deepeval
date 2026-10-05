# evals/judge.py
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge wrapper for ChatGroq models."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0):
        self.model_name_str = model_name
        self.chat_model = ChatGroq(model=model_name, temperature=temperature)
        super().__init__(model_name=model_name)

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str) -> str:
        res = self.chat_model.invoke(prompt)
        return str(res.content)

    async def a_generate(self, prompt: str) -> str:
        res = await self.chat_model.ainvoke(prompt)
        return str(res.content)

    def get_model_name(self) -> str:
        return self.model_name_str
