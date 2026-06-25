"""OpenAI-compatible VLM client — the only implementation needed.

Talks /v1/chat/completions with a base64 image part. Works against Ollama, MLX
(mlx-vlm server), and vLLM unchanged. Requires the `vlm` extra (httpx) at call time;
constructing it is dependency-free so the walking skeleton can wire it without httpx.
"""

from __future__ import annotations

import base64


class OpenAICompatVLM:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed-locally",
                 timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def read(self, image_png: bytes, prompt: str, **opts) -> str:
        import httpx  # deferred: only needed when actually calling the model

        b64 = base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            "temperature": opts.get("temperature", 0.0),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
