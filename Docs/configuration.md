# Configuration & the job API

Two ways to configure the service, with one source of truth:

- **`config.toml`** (copy from [`config.example.toml`](../config.example.toml)) — read once at
  startup. The persistent configuration.
- **`PATCH /config`** — change the output-affecting tuning knobs on a *running* service,
  in-process. Not written back to `config.toml`; restart re-reads the file.

The allowlist of what is surfaceable and what is settable lives in one place —
[`src/fusion_ocr/settings.py`](../src/fusion_ocr/settings.py), the registry both the API
and this table are generated from. `GET /config` returns every row below (secrets masked).

## Settings

`[run]` section — pipeline behaviour:

| Setting | Default | Configurable | What it does |
| --- | --- | --- | --- |
| `airgap` | `true` | **read-only** | Sealed, no-egress tier: the process refuses every non-loopback connection and DNS lookup. Surfaced but never settable over HTTP — unsealing the sensitive tier from the network would be a footgun. Change it in `config.toml` and restart. |
| `in_dir` | `"in"` | **read-only** | Drop folder the watcher scans. Identity-critical (jobs are keyed off it), so not runtime-settable. |
| `out_dir` | `"out"` | **read-only** | Where artifacts and the job DB live (`out/<sha256>/`). Identity-critical, so not runtime-settable. |
| `granularity` | `"line"` | `line` \| `word` | Overlay box granularity. `line` writes one invisible string per segment box (MVP); `word` subdivides each box across its words (follow-on). |
| `overlay_font` | `""` | path | TTF used for the invisible overlay text. A Unicode TTF is **required** for non-Latin scripts to be searchable (base-14 fonts can't encode Thai/CJK/Arabic). `""` = auto-detect (macOS Arial Unicode / common Noto paths). |
| `prefer_apple_vision` | `false` | bool | On macOS, use Apple Vision as the deterministic geometry engine for supported scripts instead of PaddleOCR — sub-2s, on-device, no server (ideal for the airgap tier and clean printed text). |
| `apple_vision_skip_vlm` | `0.92` | `0.0`–`1.0` | When a page's mean Apple Vision confidence is ≥ this, skip the VLM read — Vision's text *is* the reading (the cheap tier). Harder pages still fall through to the VLM. |
| `table_vlm_read` | `true` | bool | Route detected table regions on **scanned** pages to a focused VLM table read (crop + table prompt). Geometry still comes from the deterministic grid; this only supplies clean cell content. Born-digital tables are left to the exact text layer. |
| `fuse_min_sim` | `0.34` | `0.0`–`1.0` | Fusion anti-misalignment gate. Needleman–Wunsch always pairs a detected cluster with *some* VLM line; below this det↔VLM similarity the aligned line is treated as a misalignment, not a correction. |
| `fuse_det_conf_trust` | `0.80` | `0.0`–`1.0` | The other half of the gate: only *refuse* a dissimilar line when the detector was at least this confident. This is what protects the handwriting path — garbled `det_text` at low confidence never overrides the VLM read, which there is the truth. |
| `move_processed` | `true` | bool | Watcher moves a handled file to `in/processed/<sha>.pdf` (success) or `in/failed/<sha>.pdf` (error), so the drop folder doesn't accumulate and re-hash on every scan. **Loop only** — `--once` never moves, so a manual re-run doesn't disturb the folder. |
| `max_upload_mb` | `50` | `≥ 1.0` | `POST /jobs` rejects an upload larger than this with **413**, streamed and checked *before* the body is hashed or processed (a non-PDF body is **415**). |

`[vlm]` section — the reader endpoint (the runtime is a free variable):

| Setting | Default | Configurable | What it does |
| --- | --- | --- | --- |
| `vlm.model` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | str | Default generalist reader model name passed to the OpenAI-compatible endpoint. |
| `vlm.base_url` | `http://localhost:8080/v1` | str | Reader endpoint. Ollama / MLX / in-VPC vLLM all speak this API, so moving local → GPU is a `base_url` change. (Under airgap it must be loopback, or the seal refuses it.) |
| `vlm.api_key` | `not-needed-locally` | str (**masked**) | API key for the endpoint. Surfaced as `***` by `GET /config`. |
| `vlm.escalate_below` | `0.0` | `0.0`–`1.0` | Confidence-gated escalation: re-read a page with `escalation_model` when its mean PaddleOCR confidence is below this (or the primary read looks like a refusal). `0.0` disables it. |
| `vlm.escalation_model` | `""` | str | The stronger model to escalate to (e.g. a bigger MoE). |
| `vlm.escalation_base_url` | `""` | str | Endpoint for the escalation model (`""` reuses `vlm.base_url`). |

`[routing]` section:

| Setting | Default | Configurable | What it does |
| --- | --- | --- | --- |
| `routes` | `{}` | **read-only** | Per-script routing overrides — `{script: {paddle_lang, vlm_model, vlm_base_url}}`. A nested mapping; surfaced read-only, edit it in `config.toml`. See [routing.md](routing.md). |

### Resume interaction (why runtime changes are safe)

Every output-affecting setting above is part of the **recipe fingerprint** (`pipeline.py`)
that keys the resume cache. So a `PATCH /config` that changes, say, `fuse_min_sim`
re-keys the cache: the next job on a previously-seen PDF **reprocesses** with the new value
rather than silently returning a stale result. The read-only fields are either security
(`airgap`) or identity (`in_dir`/`out_dir`/`routes`) and don't belong on a live HTTP path.
`move_processed` and `max_upload_mb` are settable but govern ingest/ops, not OCR output, so
they're deliberately **not** fingerprinted.

## The job + config API (`api` extra)

Run it with `uvicorn fusion_ocr.api:app`. The same contract whether it runs on a desktop
now or in a VPC later.

**Auth (required, fail-closed).** Set `FUSION_OCR_API_TOKEN` in the environment — the API
**refuses to start without it**. Every request must carry `Authorization: Bearer <token>`
or it's **401** (constant-time compare). The token is env-only — never put it in
`config.toml`. The watcher and CLI need no token; they don't go through HTTP.

**Concurrency.** `POST /jobs` runs the (synchronous) pipeline off the event loop, so a long
job doesn't block other requests such as `GET /jobs` status polls.

| Method & path | Body / params | Returns |
| --- | --- | --- |
| `POST /jobs` | multipart `pdf`; query `force`, `rerun_from` | `{sha256, status}` |
| `GET /jobs/{sha256}` | — | `{sha256, status, error, artifacts}` |
| `GET /config` | — | `{settings: [{path, value, settable, kind, min?, max?, choices?, help?}, …]}` |
| `PATCH /config` | `{path: value, …}` | `{path: value, …}` (new values, secrets masked) |
| `POST /config/save` | — | `{saved: <path>}` |

`POST /jobs` streams the upload to disk in chunks (never the whole body in memory), rejecting
a non-PDF with **415** and one over `max_upload_mb` with **413** before it's hashed.

`PATCH /config` changes are **in-process only** — a restart re-reads `config.toml`. To make
the current config (including any runtime tuning) the on-disk default, call `POST
/config/save` explicitly; this opt-in step means a transient experiment can't silently
become permanent. It writes a **generated** TOML file (hand-written comments are not
preserved — `config.example.toml` stays the documented reference).

`PATCH /config` validates the **whole** body before applying anything (all-or-nothing) and
returns HTTP 400 with a `detail` message for an unknown setting, a read-only setting, or an
out-of-range / wrong-type value. Examples:

```bash
auth="authorization: Bearer $FUSION_OCR_API_TOKEN"

# see everything (api_key comes back masked)
curl -s -H "$auth" localhost:8000/config | jq '.settings[] | {path, value, settable}'

# tune the fusion gate on the running service
curl -s -X PATCH localhost:8000/config -H "$auth" \
  -H 'content-type: application/json' \
  -d '{"fuse_min_sim": 0.45, "fuse_det_conf_trust": 0.85}'

# the footgun is refused
curl -s -X PATCH localhost:8000/config -H "$auth" -d '{"airgap": false}'
# -> 400 {"detail": "'airgap' is read-only (surfaced but not configurable)"}
```
