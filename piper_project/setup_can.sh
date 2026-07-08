#!/usr/bin/env bash
set -euo pipefail

CAN_IFACE="${1:-can0}"
BITRATE="${2:-1000000}"

echo "Configuring ${CAN_IFACE} with bitrate ${BITRATE}..."

sudo -v
sudo ip link set "${CAN_IFACE}" down
sudo ip link set "${CAN_IFACE}" type can bitrate "${BITRATE}"
sudo ip link set "${CAN_IFACE}" up

echo "${CAN_IFACE} is up."
