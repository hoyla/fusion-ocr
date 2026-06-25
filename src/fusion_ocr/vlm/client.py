"""The VLM client interface and factory.

The ENTIRE local-now / CUDA-later story lives here: program against `VLMClient`,
configure `vlm.base_url`. Ollama, MLX (mlx-vlm), and vLLM all expose the same
OpenAI-compatible /v1/chat/completions with image content, so moving the heavy work
onto the transcription GPU is an endpoint change, not a code change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import Config


@runtime_checkable
class VLMClient(Protocol):
    def read(self, image_png: bytes, prompt: str, **opts) -> str:
        """Return the model's text reading of one image crop."""
        ...


def get_client(cfg: Config) -> VLMClient:
    """Construct the configured client. Only the OpenAI-compatible impl exists;
    it covers every backend we care about."""
    from .openai_compat import OpenAICompatVLM

    return OpenAICompatVLM(
        base_url=cfg.vlm.base_url,
        model=cfg.vlm.model,
        api_key=cfg.vlm.api_key,
    )
