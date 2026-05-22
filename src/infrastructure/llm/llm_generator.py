from typing import List, Dict, Any
from openai import OpenAI

from src.config.config import settings


class MedicalLLMGenerator:
    """
    Concrete LLM generator backed by a vLLM-served Qwen model.
    Reads connection details from ``src.config.config.settings`` so every
    component in the project shares a single source of truth.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.api_base = api_base or settings.LLM_BASE_URL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS

        self.client = OpenAI(
            base_url=self.api_base,
            api_key="local-no-key-required",
        )

    SYSTEM_PROMPT = (
        "You are a professional and cautious Medical AI Assistant.\n\n"
        "YOUR TASK: Answer the user's question BASED ENTIRELY on the "
        "PROVIDED DOCUMENTS below.\n\n"
        "STRICT RULES:\n"
        "1. DO NOT fabricate information, medications, or treatments "
        "that are not in the documents.\n"
        "2. If the documents lack sufficient information, ONLY output: 'I could not find sufficient medical evidence in the documents.' Do not generate any other text."
        "to answer, say clearly: 'I could not find sufficient medical "
        "evidence in the documents to answer this question.'\n"
        "3. Always remind the user to consult a real doctor.\n"
        "4. Cite the relevant condition/section when possible.\n"
        "5. Keep the answer clear, structured, and concise."
    )

    @staticmethod
    def _build_context_block(context_documents: List[Dict[str, Any]]) -> str:
        """Format retrieved docs into a numbered reference block."""
        parts: list[str] = []
        for i, doc in enumerate(context_documents, 1):
            meta = doc.get("metadata", {})
            header = ""
            if meta.get("condition"):
                header += f"Condition: {meta['condition']}"
            if meta.get("section"):
                header += f" | Section: {meta['section']}"
            parts.append(f"--- DOC {i} ({header.strip()}) ---\n{doc['content']}")
        return "\n\n".join(parts)

    def _build_messages(
        self, query: str, context_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        context_block = self._build_context_block(context_documents)
        user_prompt = (
            f"Based on the following documents, answer this question: "
            f"'{query}'\n\n"
            f"Documents:\n{context_block}"
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def generate(self, query: str, contexts: List[str]) -> str:
        """
        Implements ``LLMGenerator.generate`` (domain ABC).
        Accepts plain-string contexts and wraps them as dicts internally.
        """
        docs = [{"content": c} for c in contexts]
        return self.generate_answer(query, docs)

    def generate_answer(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        temperature: float | None = None,
    ) -> str:
        messages = self._build_messages(query, context_documents)
        temp = temperature if temperature is not None else 0.1 

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=self.max_tokens,
                presence_penalty=0.5,
                frequency_penalty=0.5,
                stop=[
                    "<|im_end|>", 
                    "<|endoftext|>", 
                    "User:", 
                    "Assistant:", 
                    "Observation:", 
                    "I could not find"
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error] Cannot connect to model at {self.api_base}: {e}"