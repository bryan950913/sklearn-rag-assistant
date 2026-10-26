"""Hardware-aware local Qwen generation with post-return elapsed-time validation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_ms: float
    backend: str


class LLMService:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        max_new_tokens: int = 128,
        timeout_seconds: int = 180,
        random_seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.random_seed = int(random_seed)
        self._tokenizer = None
        self._model = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        kwargs = {"torch_dtype": torch.float16} if self._device == "cuda" else {"torch_dtype": torch.float32}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.to(self._device)
        self._model.eval()

    @property
    def device(self) -> str:
        self.load()
        return str(self._device)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        import torch

        self.load()
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": "Answer only as the configured grounded assistant."},
            {"role": "user", "content": prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._tokenizer(
            [rendered], return_tensors="pt", truncation=True, max_length=4096
        ).to(self._device)
        selected_temperature = self.temperature if temperature is None else float(temperature)
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if selected_temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": selected_temperature, "top_p": 0.9})
        else:
            generation_kwargs["do_sample"] = False
        with torch.inference_mode():
            generated = self._model.generate(**model_inputs, **generation_kwargs)
        new_tokens = generated[:, model_inputs["input_ids"].shape[1] :]
        text = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        elapsed_ms = (time.perf_counter() - started) * 1000
        # This validates elapsed time after model.generate() returns. It is not a
        # hard interrupt and intentionally avoids fragile cross-platform workers.
        if elapsed_ms > self.timeout_seconds * 1000:
            raise TimeoutError("Local generation exceeded the configured maximum elapsed-time validation threshold")
        return GenerationResult(text=text, latency_ms=elapsed_ms, backend=self.model_name)
