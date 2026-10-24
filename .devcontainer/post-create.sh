#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[api,ocr,vlm,dev]" -c constraints.txt

if [[ ! -f config.toml ]]; then
    cp .devcontainer/config.toml config.toml
else
    echo "Preserved existing config.toml; set run.airgap=false and vlm.base_url=http://host.docker.internal:8080/v1 for container use."
fi
