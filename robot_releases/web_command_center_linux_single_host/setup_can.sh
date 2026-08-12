#!/usr/bin/env bash
set -euo pipefail

CAN_IFACE="${1:-can0}"
BITRATE="${2:-1000000}"

echo "Configuring ${CAN_IFACE} with bitrate ${BITRATE}..."

if ! command -v ip >/dev/null 2>&1; then
    echo "错误：未找到 ip 命令，请安装 iproute2：sudo apt install iproute2" >&2
    exit 127
fi

if ! ip link show "${CAN_IFACE}" >/dev/null 2>&1; then
    echo "错误：CAN 接口 ${CAN_IFACE} 不存在。请先插入 USB-CAN 设备，并用 ip link show 检查接口名称。" >&2
    exit 1
fi

sudo -v
sudo ip link set "${CAN_IFACE}" down || true
sudo ip link set "${CAN_IFACE}" type can bitrate "${BITRATE}"
sudo ip link set "${CAN_IFACE}" up

echo "${CAN_IFACE} is up."
ip -details link show "${CAN_IFACE}"
