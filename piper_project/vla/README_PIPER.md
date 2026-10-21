# LingBo-VLA 2.0 Piper WebSocket 客户端

本地客户端使用 LingBot 官方 WebSocket 协议（MessagePack）发送 observation。注意：这不再
是 `image_b64/state/instruction` 的 JSON POST。连接建立后，服务端先发送一次 metadata（你
之前看到的 `b'\\x80'` 解码后通常是空字典 `{}`），客户端读取后才发送推理 observation。

默认请求字段为：

```python
{
  "image": {
    "base_0_rgb": np.ndarray(shape=(1, 3, H, W), dtype=np.uint8)
  },
  "state": np.ndarray(shape=(1, 7), dtype=np.float32),
  "prompt": ["抓取目标物体"]
}
```

其中 `state[0]` 固定为 `[j1, j2, j3, j4, j5, j6, gripper]`。六个关节默认使用弧度，
夹爪默认使用 Piper 标定后的毫米命令。当前只采集 `camera_top`，在协议层映射为
`image.base_0_rgb`，不创建或上传腕部相机占位图。若云端使用其他图像 key，可通过
`--image-key` 修改。

云端通过同一条 WebSocket 连接返回 MessagePack 对象，动作字段需包含以下任一名称：
`action_chunk`、`action`、`actions` 或 `pred_action`。推荐结构为：

```python
{
  "action_chunk": np.ndarray(shape=(T, 6 or 7), dtype=np.float32)
}
```

每行是 `[j1..j6, gripper]`。也允许六维纯机械臂动作，此时不控制夹爪。云端应先
根据 LingBo normalization stats 反归一化，再返回实际的弧度和毫米值。若模型使用
增量关节动作，把本地配置的 `action_semantics` 改为 `delta`，并确保云端训练配置的
`subtract_state` 与之相符。

云端 LingBo 仓库使用 [robot_config_piper.yaml](robot_config_piper.yaml) 作为
`configs/robot_configs/piper.yaml`。对应 VLA 配置至少声明：

```yaml
data:
  data_name: piper
  joints:
    - arm.position: 6
    - effector.position: 1
  cameras:
    - camera_top
```

云服务器需要在 `127.0.0.1:9330` 监听 LingBot WebSocket 服务（路径由服务端约定，客户端默认不
追加路径）。在本地建立 SSH 隧道，把云端回环端口映射到本机：

```bash
ssh -N -L 9330:127.0.0.1:9330 用户名@云服务器IP
```

保持该终端运行，再开启另一个终端启动本地客户端。客户端连接的是本机地址，用户名和
密码只用于 SSH 登录，不写入 WebSocket JSON：

```bash
cd /home/jasonzhu/projects/roboticarm/piper_project
/home/jasonzhu/miniconda3/envs/piper/bin/python -m vla.client \
  --ws-url ws://127.0.0.1:9330 \
  --instruction "抓取桌面上的目标物体"
```

如果要临时连接旧的 JSON WebSocket 服务，显式指定：

```bash
python -m vla.client --protocol json --ws-url ws://127.0.0.1:9330
```

如果云端部署要求每个 episode 先 reset，可增加：

```bash
python -m vla.client --reset --robot-name piper --ws-url ws://127.0.0.1:9330
```

先运行 dry-run；它连接 CAN 并读取反馈，但不会使能或下发动作。确认云端返回的动作单位
后，再增加 `--execute`。如果服务端要求路径，例如 `/ws/infer`，将 URL 写成
`ws://127.0.0.1:9330/ws/infer`。

如果 WebSocket 服务本身启用了 HTTP Basic Auth，可通过环境变量或命令行传入。这与 SSH
登录凭据是两套独立认证：

```bash
export LINGBO_API_USER='robot-client'
export LINGBO_API_PASSWORD='服务器给出的密码'
python -m vla.client --ws-url ws://127.0.0.1:9330
```

也可以使用 `--ws-user` 和 `--ws-password`。若服务端使用 token 而不是 Basic Auth，需要
按服务端约定把 `Authorization` header 传给 `WebSocketCloudClient`。

```bash
cd /home/jasonzhu/projects/roboticarm/piper_project
/home/jasonzhu/miniconda3/envs/piper/bin/python -m vla.client \
  --ws-url ws://127.0.0.1:9330 \
  --instruction "抓取桌面上的目标物体"
```

确认打印的七维状态、云端 action chunk、关节弧度和夹爪毫米值正确后，低速执行：

```bash
/home/jasonzhu/miniconda3/envs/piper/bin/python -m vla.client \
  --ws-url ws://127.0.0.1:9330 \
  --instruction "抓取桌面上的目标物体" \
  --speed-percent 5 \
  --execute
```

本地对每行只执行简单保护：拒绝错误维度和非有限值、裁剪至 Piper 关节范围、限制
相邻动作的关节变化和夹爪变化。模型返回的整个 action chunk 会依次执行，不使用本地
抓取状态机。

## 官方输入格式探测脚本

可使用独立脚本直接测试 LingBot 官方 MessagePack 输入，不经过 Piper 客户端的
`image_b64` JSON 协议：

```bash
cd /home/jasonzhu/projects/roboticarm/piper_project
/home/jasonzhu/miniconda3/envs/piper/bin/python \
  -m vla.deploy.test_lingbot_official_ws \
  --url ws://127.0.0.1:9330 \
  --state-dim 8
```

脚本发送 `image.base_0_rgb`（`(1,3,224,224)`、`uint8`）、`state`（`(1,state_dim)`、
`float32`）和 `prompt`。可以将 `--state-dim` 改为 7、8、14、16 分别测试 checkpoint
可能使用的状态维度；如果服务端要求 episode 初始化，可增加 `--reset --robot-name piper`。

输出中的 `metadata`、`reply wire type`、`reply decoded` 和 `reply keys` 用于判断服务端
是否返回了动作。

## 只采集并检查本地 observation

如果暂时只想确认相机和 Piper 状态，不连接云端推理，可以执行：

```bash
/home/jasonzhu/miniconda3/envs/piper/bin/python \
  -m vla.client \
  --capture-only \
  --image-key base_0_rgb \
  --instruction "测试输入"
```

客户端会打印：

```text
observation fields: ['image', 'state', 'prompt']
image fields: ['base_0_rgb']
image shape/dtype/min/max: (1, 3, H, W) uint8 ...
state shape/dtype/values: (1, 7) float32 ...
prompt: ['测试输入']
```

同时保存原始相机图像和 LingBot 数组：

```bash
mkdir -p /tmp/piper_vla_debug

/home/jasonzhu/miniconda3/envs/piper/bin/python \
  -m vla.client \
  --capture-only \
  --dump-image /tmp/piper_vla_debug/camera_top.jpg \
  --dump-observation /tmp/piper_vla_debug/observation.npz
```

检查保存的数组：

```bash
/home/jasonzhu/miniconda3/envs/piper/bin/python - <<'PY'
import numpy as np

data = np.load('/tmp/piper_vla_debug/observation.npz')
print('image:', data['image'].shape, data['image'].dtype)
print('state:', data['state'].shape, data['state'].dtype, data['state'])
print('prompt:', data['prompt'])
PY
```

`--capture-only` 会在采集和打印后退出，不会连接 WebSocket，也不会执行机械臂动作。

## 单头部相机映射到三路图像字段

当前只有一个 `camera_top` 时，客户端默认开启复制映射：同一帧图像会同时放入：

```python
{
    "image": {
        "base_0_rgb": frame,
        "left_wrist_0_rgb": frame,
        "right_wrist_0_rgb": frame,
    }
}
```

这只是为了满足服务端的字段结构，不代表获得了真实腕部视角。配置项为：

```yaml
mirror_wrist_images: true
```

也可以通过命令行控制：

```bash
# 开启复制（默认）
python -m vla.client --mirror-wrist-images

# 关闭复制，只发送 base_0_rgb
python -m vla.client --no-mirror-wrist-images
```

只有在云端明确要求三路图像字段、且允许占位/重复图像时才开启。若模型使用真实多相机
训练，复制头部图像可能造成分布偏移，最终仍建议增加真实左右腕部相机。

## 验证本地发送字节和 observation schema

可以保存客户端准备发送的数组和完整 MessagePack 字节：

```bash
mkdir -p /tmp/piper_vla_debug
/home/jasonzhu/miniconda3/envs/piper/bin/python \
  -m vla.client \
  --capture-only \
  --mirror-wrist-images \
  --dump-observation /tmp/piper_vla_debug/observation.npz \
  --dump-msgpack /tmp/piper_vla_debug/request.msgpack
```

然后做本地 schema 和 MessagePack round-trip 检查：

```bash
/home/jasonzhu/miniconda3/envs/piper/bin/python \
  -m vla.deploy.validate_lingbot_observation \
  --npz /tmp/piper_vla_debug/observation.npz \
  --msgpack /tmp/piper_vla_debug/request.msgpack \
  --state-dim 7
```

该检查会验证：顶层字段、三个图像 key、`(1,3,H,W)`、`uint8`、`state (1,D)`、
`float32`、`prompt list[str]`，并验证 MessagePack 编码后再解码仍保持相同结构。

要验证“云端实际接受的格式”，还必须在云端 WebSocket handler 中对同一字节包执行：

```python
observation = unpackb(raw_message, raw=False)
print(observation.keys())
print(observation["image"].keys())
print(observation["state"].shape, observation["state"].dtype)
print(observation["prompt"])
```

本地 round-trip 只能证明编码正确，不能证明云端模型配置允许该状态维度或图像 key。
