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
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen3_5 import modeling_qwen3_5

from cutamp.config import TAMPConfiguration

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def strip_thinking_tokens(text: str) -> str:
    """Remove Qwen-style thinking traces from generated text."""
    return _THINK_TAG_PATTERN.sub("", text).strip()


def configure_qwen3_5_runtime() -> None:
    """Force Qwen3.5 to use the built-in torch gated-delta implementation."""
    modeling_qwen3_5.chunk_gated_delta_rule = None
    modeling_qwen3_5.fused_recurrent_gated_delta_rule = None
    modeling_qwen3_5.FusedRMSNormGated = None
    modeling_qwen3_5.is_fast_path_available = False


class BaseVLMClient(ABC):
    """Abstract multimodal client interface."""

    @abstractmethod
    def generate(self, prompt: str, image_path: str | None = None, stage: str = "generic") -> str:
        """Generate a response for the given prompt and optional image."""


class TransformersVLMClient(BaseVLMClient):
    """Transformers-backed multimodal VLM client."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        device_map: str | None = None,
        attention_implementation: str | None = None,
        quantization: str = "none",
        max_new_tokens: int = 512,
        max_time_sec: float | None = None,
        temperature: float = 0.0,
        do_sample: bool = False,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.device_map = device_map
        self.attention_implementation = attention_implementation
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.max_time_sec = max_time_sec
        self.temperature = temperature
        self.do_sample = do_sample
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.torch_dtype = _TORCH_DTYPES[self.dtype]
        configure_qwen3_5_runtime()
        self.processor = AutoProcessor.from_pretrained(self.model_name, **self._processor_kwargs())
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **self._model_kwargs())
        if self.device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        self.model_device = next(iter(self.model.parameters())).device
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.pad_token_id = self.processor.tokenizer.pad_token_id or self.eos_token_id

    def _processor_kwargs(self) -> dict:
        kwargs = {"trust_remote_code": True}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        return kwargs

    def _quantization_config(self) -> BitsAndBytesConfig | None:
        if self.quantization == "none":
            return None
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _model_kwargs(self) -> dict:
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        if self.attention_implementation is not None:
            kwargs["attn_implementation"] = self.attention_implementation
        quantization_config = self._quantization_config()
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        elif self.quantization != "none":
            kwargs["device_map"] = "auto"
        return kwargs

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

    def _cached_response(self, cache_key: str) -> str | None:
        if self.cache_dir is None:
            return None
        cache_path = self.cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["response"]

    def _save_cached_response(
        self,
        cache_key: str,
        prompt: str,
        response: str,
        stage: str,
        image_path: str | None,
    ) -> None:
        if self.cache_dir is None:
            return
        cache_path = self.cache_dir / f"{cache_key}.json"
        payload = {
            "model_name": self.model_name,
            "stage": stage,
            "image_path": image_path,
            "prompt": prompt,
            "response": response,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _build_messages(self, prompt: str, image_path: str | None) -> list[dict]:
        content = []
        if image_path is not None:
            content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, prompt: str, image_path: str | None) -> dict:
        messages = self._build_messages(prompt, image_path)
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return dict(batch.to(self.model_device).items())

    def _generation_kwargs(self) -> dict:
        kwargs = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.pad_token_id,
        }
        if self.eos_token_id is not None:
            kwargs["eos_token_id"] = self.eos_token_id
        if self.max_time_sec is not None:
            kwargs["max_time"] = self.max_time_sec
        if self.do_sample:
            kwargs["temperature"] = self.temperature
        return kwargs

    def generate(self, prompt: str, image_path: str | None = None, stage: str = "generic") -> str:
        cache_key = self._cache_key(prompt, image_path, stage)
        cached = self._cached_response(cache_key)
        if cached is not None:
            return cached

        model_inputs = self._prepare_inputs(prompt, image_path)
        generation_kwargs = self._generation_kwargs()
        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generation_kwargs)

        prompt_ids = model_inputs.get("input_ids")
        generated_ids = output_ids
        if prompt_ids is not None:
            generated_ids = [row[len(prefix) :] for prefix, row in zip(prompt_ids, output_ids)]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        response = strip_thinking_tokens(response)
        self._save_cached_response(cache_key, prompt, response, stage, image_path)
        return response


def create_vlm_client(config: TAMPConfiguration) -> BaseVLMClient:
    """Create a VLM client from the TAMP configuration."""
    return TransformersVLMClient(
        model_name=config.vlm_model_name,
        device=config.vlm_device,
        dtype=config.vlm_dtype,
        device_map=config.vlm_device_map,
        attention_implementation=config.vlm_attention_implementation,
        quantization=config.vlm_quantization,
        max_new_tokens=config.vlm_max_new_tokens,
        max_time_sec=config.vlm_max_time_sec,
        temperature=config.vlm_temperature,
        do_sample=config.vlm_do_sample,
        cache_dir=config.vlm_cache_dir,
    )
