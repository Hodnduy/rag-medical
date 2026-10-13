import re
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
            timeout=10.0,
        )
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._mlx_available = False
        try:
            import mlx_lm  # noqa: F401
            self._mlx_available = True
        except ImportError:
            pass

    SYSTEM_PROMPT = (
        "You are a professional and cautious Medical AI Assistant.\n\n"
        "YOUR TASK: Answer the user's question BASED ENTIRELY on the "
        "PROVIDED DOCUMENTS below.\n\n"
        "STRICT RULES:\n"
        "1. DO NOT fabricate information, medications, or treatments that are not in the documents.\n"
        "2. If the documents do not contain relevant information to answer the question, state that the provided text does not contain enough information.\n"
        "3. Always remind the user to consult a real doctor.\n"
        "4. Cite the relevant condition/section when possible.\n"
        "5. Keep the answer concise, direct and to the point (maximum 150 words). Do not repeat sentences or documents.\n"
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

    @staticmethod
    def _clean_repeated_response(text: str) -> str:
        """
        Post-processing fallback to truncate repeating blocks if LLM ignores
        stop tokens or leaks special tokens like <|endoftext|>.
        """
        if not text:
            return text

        # 0. Truncate at leaked special tokens immediately
        for special_tok in ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]:
            if special_tok in text:
                text = text.split(special_tok)[0].strip()

        # 1. Handle repeated "Final Answer:" blocks
        if text.count("Final Answer:") > 1:
            sections = text.split("Final Answer:")
            for sec in sections[1:]:
                clean_sec = sec.strip()
                if clean_sec:
                    text = clean_sec
                    break

        # 2. Handle repeated end-of-response markers
        for marker in [
            "--- End of Response ---",
            "--- End of Documents ---",
            "--- END ---",
        ]:
            if marker in text:
                text = text.split(marker)[0].strip()

        # 3. Handle leftover "Final Answer:" prefix if present
        if text.startswith("Final Answer:"):
            text = text[len("Final Answer:"):].strip()

        # 4. Remove self-check / compliance chatter lines (✅, ✓, etc.)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are only compliance checks
            if re.match(r'^[✅✓☑⬜●•]\s', stripped):
                continue
            # Skip lines like "(No fabricated information...)" or "(Source: ...)"
            if stripped.startswith("(") and stripped.endswith(")") and len(stripped) > 50:
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # 5. Handle repeating identical paragraphs
        paragraphs = text.split("\n\n")
        seen = []
        for p in paragraphs:
            p_strip = p.strip()
            if p_strip and p_strip not in seen:
                seen.append(p_strip)
        text = "\n\n".join(seen)

        # 6. Detect tail repetition: if the last N lines are short and
        #    mostly identical, truncate them
        lines = text.split("\n")
        if len(lines) > 10:
            tail = lines[-10:]
            unique_tail = set(l.strip() for l in tail if l.strip())
            if len(unique_tail) <= 2:
                # Tail is degenerate repetition – remove it
                text = "\n".join(lines[:-10]).strip()

        return text.strip()

    def _generate_via_mlx(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """In-process MLX generation (Apple Silicon GPU) when no HTTP server is running."""
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        if self._mlx_model is None:
            print(f"[MedicalLLMGenerator] Loading '{self.model_name}' via MLX (Metal GPU) …")
            self._mlx_model, self._mlx_tokenizer = mlx_lm.load(self.model_name)

        prompt = self._mlx_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=temperature) if temperature > 0 else None
        gen_kwargs = {"sampler": sampler} if sampler is not None else {}

        raw_content = mlx_lm.generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
            **gen_kwargs,
        )
        return self._clean_repeated_response(raw_content)

    def generate_answer(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        temperature: float | None = None,
    ) -> str:
        messages = self._build_messages(query, context_documents)
        temp = temperature if temperature is not None else self.temperature

        # 1. Try OpenAI-compatible HTTP server (vLLM / mlx_lm server / Ollama)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens,
                presence_penalty=0.0,
                frequency_penalty=0.3,
                stop=[
                    "<|im_end|>", 
                    "<|endoftext|>", 
                    "User:", 
                    "Assistant:", 
                    "Observation:",
                    "--- END ---",
                    "--- End of Response ---",
                    "--- End of Documents ---",
                    "Final Answer:"
                ],
                extra_body={
                    "stop_token_ids": [151643, 151645]
                }
            )
            raw_content = response.choices[0].message.content.strip()
            return self._clean_repeated_response(raw_content)
        except Exception as http_err:
            # 2. If HTTP server is unreachable, fallback to in-process MLX on macOS
            if self._mlx_available:
                try:
                    return self._generate_via_mlx(messages, temp)
                except Exception as mlx_err:
                    return f"[LLM Error] HTTP error: {http_err} | MLX error: {mlx_err}"
            return f"[LLM Error] Cannot connect to model at {self.api_base}: {http_err}"