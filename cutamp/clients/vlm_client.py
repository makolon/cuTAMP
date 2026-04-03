# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""VLM client implementations used by the VLM-TAMP pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Iterable, Optional

from cutamp.config import TAMPConfiguration

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking_tokens(text: str) -> str:
    """Remove Qwen-style thinking traces from the generated output."""
    return _THINK_TAG_PATTERN.sub("", text).strip()


class BaseVLMClient(ABC):
    """Abstract multimodal client interface."""

    @abstractmethod
    def generate(self, prompt: str, image_path: str | None = None, stage: str = "generic") -> str:
        """Generate a response for the given prompt and optional image."""


class StubVLMClient(BaseVLMClient):
    """
    Lightweight stub client for testing.

    Responses can be specified as:
    - a flat iterable used as a global queue
    - a mapping from stage name to iterables
    - a callable `(prompt, image_path, stage) -> str`
    """

    def __init__(
        self,
        responses: (
            Iterable[str]
            | dict[str, Iterable[str]]
            | Callable[[str, str | None, str], str]
            | None
        ) = None,
    ):
        self._callable = responses if callable(responses) else None
        self._global_queue = deque()
        self._stage_queues: dict[str, deque[str]] = defaultdict(deque)

        if responses is None or callable(responses):
            return

        if isinstance(responses, dict):
            for stage, stage_responses in responses.items():
                if isinstance(stage_responses, str):
                    stage_responses = [stage_responses]
                self._stage_queues[stage].extend(stage_responses)
        else:
            if isinstance(responses, str):
                responses = [responses]
            self._global_queue.extend(responses)

    def generate(self, prompt: str, image_path: str | None = None, stage: str = "generic") -> str:
        if self._callable is not None:
            return str(self._callable(prompt, image_path, stage))
        if self._stage_queues[stage]:
            return self._stage_queues[stage].popleft()
        if self._global_queue:
            return self._global_queue.popleft()
        raise RuntimeError(f"No stub VLM response available for stage '{stage}'")


class TransformersVLMClient(BaseVLMClient):
    """Transformers-backed multimodal VLM client with optional response caching."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._processor = None
        self._model = None
        self._torch = None
        self._pil_image = None
        self._model_device = None

    def _ensure_loaded(self):
        if self._processor is not None and self._model is not None:
            return

        import torch
        from PIL import Image
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as _AutoModel
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as _AutoModel
            except ImportError:
                from transformers import AutoModelForCausalLM as _AutoModel

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.dtype]

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = _AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        self._torch = torch
        self._pil_image = Image
        self._model_device = torch.device(self.device)

    def _cache_key(self, prompt: str, image_path: str | None, stage: str) -> str:
        payload = {
            "model_name": self.model_name,
            "prompt": prompt,
            "image_path": image_path,
            "stage": stage,
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        if image_path is not None:
            digest.update(Path(image_path).read_bytes())
        return digest.hexdigest()

    def _read_cache(self, cache_key: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["response"]

    def _write_cache(self, cache_key: str, prompt: str, response: str, stage: str, image_path: str | None) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{cache_key}.json"
        payload = {
            "model_name": self.model_name,
            "stage": stage,
            "image_path": image_path,
            "prompt": prompt,
            "response": response,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _prepare_inputs(self, prompt: str, image_path: str | None):
        self._ensure_loaded()
        assert self._processor is not None
        assert self._pil_image is not None

        image = None
        if image_path is not None:
            image = self._pil_image.open(image_path).convert("RGB")

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        if hasattr(self._processor, "apply_chat_template"):
            try:
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if image is not None:
                inputs = self._processor(images=image, text=text, return_tensors="pt")
            else:
                inputs = self._processor(text=text, return_tensors="pt")
        else:
            if image is not None:
                inputs = self._processor(images=image, text=prompt, return_tensors="pt")
            else:
                inputs = self._processor(text=prompt, return_tensors="pt")
        return inputs

    def generate(self, prompt: str, image_path: str | None = None, stage: str = "generic") -> str:
        cache_key = self._cache_key(prompt, image_path, stage)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        self._ensure_loaded()
        assert self._torch is not None
        assert self._model is not None

        inputs = self._prepare_inputs(prompt, image_path)
        inputs = {k: v.to(self._model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, **generation_kwargs)

        prompt_tokens = inputs.get("input_ids")
        if prompt_tokens is not None:
            generated = outputs[:, prompt_tokens.shape[-1] :]
        else:
            generated = outputs
        response = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        response = strip_thinking_tokens(response)
        self._write_cache(cache_key, prompt, response, stage, image_path)
        return response


def create_vlm_client(config: TAMPConfiguration) -> BaseVLMClient:
    """Create a VLM client from the TAMP configuration."""
    if config.vlm_backend == "stub":
        return StubVLMClient()
    if config.vlm_backend == "transformers":
        return TransformersVLMClient(
            model_name=config.vlm_model_name,
            device=config.vlm_device,
            dtype=config.vlm_dtype,
            max_new_tokens=config.vlm_max_new_tokens,
            temperature=config.vlm_temperature,
            do_sample=config.vlm_do_sample,
            cache_dir=config.vlm_cache_dir,
        )
    raise ValueError(f"Unsupported VLM backend: {config.vlm_backend}")
