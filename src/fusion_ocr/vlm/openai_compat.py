"""OpenAI-compatible VLM client — the only implementation needed.

Talks /v1/chat/completions with a base64 image part. Works against Ollama, MLX
(mlx-vlm server), and vLLM unchanged. Requires the `vlm` extra (httpx) at call time;
constructing it is dependency-free so the walking skeleton can wire it without httpx.

Hardening (review_01): one keep-alive `httpx.Client` reused across pages (not one per
call); a `max_tokens` cap so a pathological page can't generate until the timeout;
retry-with-backoff on transient 5xx / transport errors; and JPEG (not PNG) image parts to
keep the base64 payload small on the wire. An `AirgapError` is surfaced immediately and
never retried — a sealed tier must fail loud, not spin.
"""

from __future__ import annotations

import base64
import time

from ..config import AirgapError, Config


def _airgap_in_chain(exc: BaseException) -> AirgapError | None:
    """An AirgapError anywhere in the cause/context chain — httpx wraps a connect-time
    OSError (which the airgap guard raises) in a TransportError, so the guard's exception
    arrives as `__cause__`, not the top-level type."""
    seen: BaseException | None = exc
    depth = 0
    while seen is not None and depth < 10:
        if isinstance(seen, AirgapError):
            return seen
        seen = seen.__cause__ or seen.__context__
        depth += 1
    return None


class OpenAICompatVLM:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed-locally",
                 timeout: float = 600.0, max_tokens: int = 4096, max_retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._http = None  # lazily-created persistent httpx.Client (keep-alive across pages)

    def _client(self):
        import httpx  # deferred: only needed when actually calling the model

        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout)
        return self._http

    def read(self, image_bytes: bytes, prompt: str, image_format: str = "jpeg", **opts) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{b64}"},
                        },
                    ],
                }
            ],
            "temperature": opts.get("temperature", 0.0),
        }
        max_tokens = opts.get("max_tokens", self.max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return self._post_with_retry(payload, headers)

    def _post_with_retry(self, payload, headers) -> str:
        import httpx

        url = f"{self.base_url}/chat/completions"
        attempts = self.max_retries + 1
        for i in range(attempts):
            last = i == attempts - 1
            try:
                resp = self._client().post(url, json=payload, headers=headers)
                if resp.status_code >= 500 and not last:
                    time.sleep(_backoff(i))
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 — classify, then re-raise or retry
                airgap = _airgap_in_chain(exc)
                if airgap is not None:
                    raise airgap   # sealed-tier misconfig: fail loud, never retry/degrade
                if not last and isinstance(exc, httpx.TransportError):
                    time.sleep(_backoff(i))
                    continue
                raise
        return ""  # unreachable: the loop always returns or raises

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def probe(self) -> None:
        """Readiness probe: a tiny 1-token inference on a blank image. RAISES on failure.
        A plain GET /v1/models is NOT sufficient — a wedged mlx-vlm server answers that 200 while
        failing generation; only a real (if trivial) completion proves the reader can read. Doubles
        as a warm-up: it triggers the model load if the server is cold."""
        import io

        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(buf, format="JPEG")
        self.read(buf.getvalue(), "Reply with OK.", max_tokens=1)


def _backoff(attempt: int) -> float:
    return min(0.5 * (2 ** attempt), 8.0)   # 0.5s, 1s, 2s, 4s … capped at 8s


def preflight_reader(cfg: Config, timeout: float = 120.0) -> tuple[bool, str]:
    """Check the configured VLM reader can actually READ before a batch run, so a dead or wedged
    server surfaces up front instead of silently degrading every page to det_text. Returns
    ``(ok, detail)``. Bounded (default 120s) to cover a cold model load without hanging forever on
    a wedged one; loopback-only under airgap, like the reads themselves. Non-fatal by contract —
    the caller decides whether to warn or abort."""
    client = OpenAICompatVLM(base_url=cfg.vlm.base_url, model=cfg.vlm.model,
                             api_key=cfg.vlm.api_key, timeout=timeout, max_tokens=1, max_retries=1)
    try:
        client.probe()
        return True, f"reader ready: {cfg.vlm.model} @ {cfg.vlm.base_url}"
    except AirgapError as exc:
        return False, f"reader endpoint blocked by airgap: {exc}"
    except Exception as exc:  # noqa: BLE001 — any failure means "not ready"
        return False, (f"reader NOT ready ({cfg.vlm.model} @ {cfg.vlm.base_url}): "
                       f"{type(exc).__name__}: {exc}")
    finally:
        client.close()
