#!/usr/bin/env bash
# Fail fast if a self-hosted Omni CI job was scheduled onto a runner that does
# not provide an explicit 2-GPU lane through NVIDIA_VISIBLE_DEVICES.
set -euo pipefail

visible_devices="${NVIDIA_VISIBLE_DEVICES:-}"
if [[ -z "${visible_devices}" ]]; then
  echo "::error::NVIDIA_VISIBLE_DEVICES is unset. Runner must be launched as a configured h100 lane." >&2
  exit 1
fi

if [[ ! "${visible_devices}" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "::error::NVIDIA_VISIBLE_DEVICES must name exactly two numeric GPU devices, got '${visible_devices}'." >&2
  exit 1
fi

echo "Validated Omni CI 2-GPU lane: NVIDIA_VISIBLE_DEVICES=${visible_devices}"
