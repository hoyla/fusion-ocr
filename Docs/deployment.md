# Deploying behind nginx + TLS

The job API speaks plain HTTP and authenticates with a bearer token. For a real deployment,
put it behind a reverse proxy that terminates TLS, so the token and the documents travel
encrypted. This is the groundwork; sample configs are in [`deploy/`](../deploy).

```
client ──HTTPS──▶ nginx :443 ──HTTP──▶ fusion-ocr-serve 127.0.0.1:8000   (API: enqueue + status)
                 (TLS, certs)          (localhost, bearer auth)   │
                                                                   ▼ shared in/, out/, jobs.sqlite
                                        fusion-ocr (watcher)  ─────┘   (WORKER: drains the queue)
                                                                   │
                                                                   └─▶ reader (VLM) — loopback, or another host if airgap=false
```

Two processes share the estate: **`fusion-ocr-serve`** (the API — enqueues uploads and
serves status) and **`fusion-ocr`** (the watcher — the worker that drains the queue and runs
the pipeline). The backend stays on **localhost**; only nginx is exposed. nginx terminates
TLS and forwards requests (with the `Authorization` header intact) to the API, which still
enforces the token. `POST /jobs` returns `202 queued` immediately; clients poll
`GET /jobs/{sha256}` (or pull `GET /jobs?status=done`) for the result.

## 1. Backend config (`config.toml`)

```toml
[run]
api_host = "127.0.0.1"          # behind the proxy — NOT 0.0.0.0; only nginx faces the network
api_port = 8000                 # must match the upstream in the nginx config
forwarded_allow_ips = "127.0.0.1"   # trust X-Forwarded-* from the colocated proxy
max_upload_mb = 50              # nginx client_max_body_size must be >= this
```

The bearer token is **env, not config**: `FUSION_OCR_API_TOKEN` (the API won't start without
it). On Linux it lives in the systemd `EnvironmentFile`; see [`deploy/fusion-ocr.service`](../deploy/fusion-ocr.service).

Run it with `fusion-ocr-serve` (it enables `proxy_headers` so the app sees the real client IP
and the `https` scheme via `forwarded_allow_ips`).

## 2. Certificates

- **Internal / testing — self-signed** (fine on a trusted estate; clients must trust the cert):
  ```bash
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout privkey.pem -out fullchain.pem -subj "/CN=ocr.internal.example"
  ```
- **Proper CA** — Let's Encrypt (`certbot --nginx`) if the host has a public DNS name, or your
  organisation's internal CA for an on-estate name. Point `ssl_certificate` /
  `ssl_certificate_key` at the issued files.

## 3. nginx

Copy [`deploy/nginx.fusion-ocr.conf`](../deploy/nginx.fusion-ocr.conf), fill the
`<PLACEHOLDERS>` (hostname, cert paths, backend port), then `nginx -t && systemctl reload nginx`.

**Two settings that bite if you skip them:**
- `client_max_body_size` must be **≥ `max_upload_mb`** or nginx 413s the upload before the app
  can apply its own (more informative) limit.
- `proxy_*_timeout` only needs to cover the **upload**, not the OCR run — `POST /jobs`
  enqueues and returns `202` immediately. The sample uses modest values; bump `proxy_send`
  only for very large PDFs over slow links.

> **The queue.** Submit is asynchronous: the API writes the upload to `in/`, registers it
> `queued`, and returns. The worker (`fusion-ocr`) claims queued jobs atomically and runs
> them, so you can run more than one worker without double-processing. `JobStore` (SQLite)
> *is* the queue; its method surface is the contract a distributed queue (ElasticMQ / SQS,
> on-estate) would implement later — and artifacts are content-addressed via `storage.py`,
> the swap point for an object store (Garage / S3). Neither is needed at current volume.

> **One machine only (for now).** Multi-worker is safe **on a single machine sharing a local
> disk** — that is the whole supported surface. Do **not** point two machines at a shared
> folder: SQLite WAL does not work on network filesystems (SMB/NFS) and risks silent database
> corruption, and concurrent writers would race on `out/`. A cluster of Macs needs the queue +
> object-store adapters above; until then, the only component that can live on another machine
> is the reader (VLM) endpoint. (Review 03.)

## 4. Run as services (Linux)

You run **two** units — the API and the worker:
- [`deploy/fusion-ocr.service`](../deploy/fusion-ocr.service) — the API (`fusion-ocr-serve`).
- [`deploy/fusion-ocr-worker.service`](../deploy/fusion-ocr-worker.service) — the worker (`fusion-ocr`).

Drop the token into `/etc/fusion-ocr/fusion-ocr.env` (root-owned, `chmod 600`), then
`systemctl enable --now fusion-ocr fusion-ocr-worker`. They share `in/`, `out/`, and
`out/jobs.sqlite`. On macOS use launchd / `brew services`; in a container, run one process
per container (API and worker) against shared volumes.

> Run at least one worker, or uploads sit in the queue as `queued` forever. The atomic claim
> means you can scale to several workers later without double-processing.

## Simpler alternative: Caddy (automatic TLS)

If you'd rather not manage certs, Caddy auto-provisions and renews them. The whole config is:

```caddyfile
ocr.internal.example {
    request_body { max_size 60MB }
    reverse_proxy 127.0.0.1:8000 {
        transport http { read_timeout 600s }
    }
}
```

## Security checklist

- [ ] `FUSION_OCR_API_TOKEN` set to a strong random value; never committed, never in `config.toml`.
- [ ] Backend on `127.0.0.1` — confirm it's **not** also reachable on `0.0.0.0` (only nginx should be).
- [ ] TLS only; HTTP 301-redirects to HTTPS (the sample does this).
- [ ] `client_max_body_size` ≥ `max_upload_mb`; proxy timeouts ≥ worst-case job.
- [ ] If the reader (VLM) is on another host, `airgap = false` and `vlm.base_url` is that
      host's **IP** — otherwise keep `airgap = true` with a loopback reader.
- [ ] Firewall: only 443 (and 80 for the redirect) open to the network.
- [ ] **Sealed tier: `airgap = true` is a tripwire, not a wall.** The guard is Python-level
      (it patches `connect`/`connect_ex`/`getaddrinfo`); it does not cover
      `gethostbyname`/raw UDP, native-code sockets (paddle/onnxruntime internals), or
      subprocesses. For the most-sensitive tier, pair it with an **OS-level** control — a
      network-less user account, pf rules blocking outbound for the worker, or
      `sandbox-exec` — and treat the Python seal as defence-in-depth. (Review 03; an OS-level
      recipe is on the roadmap.)
