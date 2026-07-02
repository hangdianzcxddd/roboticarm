# catch_server.py 参数、坐标系与超限排查说明

本文对应命令：

```bash
python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --can can0 \
  --execute \
  --hand-eye-result \
  --speed 5 \
  --debug-plan
```

`catch_server.py` 的核心任务是：接收 Windows 侧发来的相机坐标点，使用手眼标定矩阵把它转换到机器人基座坐标系，再根据夹爪 TCP 偏移生成 Piper SDK 实际需要下发的 J6 位姿，最后执行接近、抓取、抬升和放置。

## 1. 坐标系关系

整体链路如下：

```text
Windows RGB 像素 (u, v)
  |
  |  D435i 内参 fx/fy/ppx/ppy + 对齐后的 depth
  v
相机坐标 P_camera = (Xc, Yc, Zc), 单位 m
  |
  |  hand_eye_result.json 中的 T_base_camera
  |  P_base = T_base_camera @ P_camera
  v
机器人基座坐标 P_base = target_tcp, 单位 mm
  |
  |  选择抓取 TCP 姿态 rx/ry/rz
  |  加 target_z_offset / approach_height / lift_height
  v
目标 TCP 位姿 target/grasp/approach/lift/place
  |
  |  官方夹爪 TCP offset: J6 -> TCP = (0, 0, 145) mm
  |  J6 = TCP - R(rx,ry,rz) @ offset
  v
Piper SDK 下发的 J6 位姿
```

关键点：

- Windows 侧只发送 `camera XYZ`，不发送机器人坐标。
- Linux 侧 `--hand-eye-result` 决定是否使用标定矩阵。如果没有这个参数，就走模拟映射，不是真实抓取。
- `target_tcp` 是夹爪指尖/抓取点希望到达的位置，不是机械臂反馈中的末端 J6 位置。
- Piper SDK 控制的是 J6 位姿，所以代码必须把 TCP 位姿转换成 J6 位姿。
- 145 mm TCP 偏移会随姿态旋转投影到 base 坐标。如果当前姿态 `ry≈66°`，offset 很可能在 X 方向投影约 130 mm，导致：

```text
target_tcp.x = 180 mm
offset_base.x ≈ 130 mm
J6.x = 180 - 130 ≈ 50 mm
```

这就是你经常看到 `target_tcp` 看起来还可以，但 `approach_j6/grasp_j6` 的 `x≈0~50 mm`，随后 SDK 报超限的原因之一。

## 2. 重要日志字段

启动时：

```text
Using hand-eye camera->base transform: .../hand_eye_result.json
Hand-eye quality: translation_rms=... rotation_rms=...
Catch args: Namespace(...)
```

含义：

- `Using hand-eye...`：正在使用 `camera -> base` 的手眼矩阵。
- `Hand-eye quality`：标定残差。`translation_rms` 如果达到二三十毫米，真实抓取会有明显偏差；`rotation_rms` 偏大时，越远的点误差越大。
- `Catch args`：本次服务所有运动和安全参数。

点击或视觉检测后，`--debug-plan` 会打印：

```text
camera: u=... v=... xyz=(...) m
base_from_hand_eye: x=... y=... z=... mm
mapping=hand_eye orientation=... target_tcp=... grasp_tcp=... place_tcp=...
tcp_offset_base: dx=... dy=... dz=... mm (J6 = TCP - offset_base)
current_j6=...
staged stage_lift=...
waypoint approach=...
waypoint grasp=...
```

看日志时建议按这个顺序判断：

1. `camera xyz` 是否合理，尤其深度 `z_m` 是否是物体深度。
2. `base_from_hand_eye` 是否落在真实桌面/物体对应的机器人基座区域。
3. `target_tcp` 是否在可抓区域，例如 X 不要太小、Y 不要太偏。
4. `tcp_offset_base` 是否把 J6 拉到了危险区域。
5. `approach/grasp/lift` 的 J6 是否满足 `x/y/z/radius` 的保守限制。

## 3. 参数总览

### 3.1 服务与执行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `host` | `0.0.0.0` | Linux 侧 TCP server 监听地址。`0.0.0.0` 表示监听所有网卡。 |
| `port` | `5005` | TCP 端口，Windows 客户端需要连接这个端口。 |
| `can` | `can0` | Piper CAN 设备名。 |
| `execute` | `False` | 是否真实执行机械臂动作。不加时只计算和返回规划。 |
| `speed` | `5` | Piper 运动速度百分比，初调建议低速。 |
| `print_feedback` | `False` | 执行前后打印机械臂反馈。 |
| `debug_plan` | `False` | 每次收到点后打印完整坐标转换和 waypoint 规划。强烈建议调试时开启。 |

### 3.2 运动模式与等待参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `approach_mode` | `P` | approach 和 staged XY 的运动模式，`P` 为 MoveP，`L` 为 MoveL。 |
| `grasp_mode` | `L` | 下探抓取 waypoint 的运动模式。 |
| `lift_mode` | `L` | 抓取后抬升和撤离的运动模式。 |
| `command_duration_s` | `1.0` | 每个 waypoint 连续发送命令的时间。 |
| `command_period_s` | `0.02` | 命令发送周期。 |
| `motion_timeout_s` | `8.0` | 发送完命令后等待到达 waypoint 的最长时间。 |
| `position_tolerance_mm` | `8.0` | 判断 waypoint 到达的位置误差阈值。 |
| `orientation_tolerance_deg` | `8.0` | 判断 waypoint 到达的姿态误差阈值。 |
| `no_wait_waypoints` | `False` | 不等待 waypoint 到达，直接进入下一步。真实抓取不建议开启。 |

### 3.3 手眼与模拟映射参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `hand_eye_result` | `None` 或默认 json | 使用手眼标定 `camera -> base` 矩阵。你的命令里已启用。 |
| `use_current_pose_as_base` | `False` | 仅用于模拟映射，用当前 TCP 当作映射中心。不能和 `hand_eye_result` 同时使用。 |
| `base_x_mm/base_y_mm/base_z_mm` | `150/0/220` | 没有手眼矩阵时的模拟映射中心。启用 `hand_eye_result` 后基本不参与真实计算。 |
| `scale_x/scale_y/scale_z` | `120/120/80` | 模拟映射比例。启用 `hand_eye_result` 后基本不参与真实计算。 |
| `max_delta_x/y/z_mm` | `40/40/30` | 模拟映射限幅。启用 `hand_eye_result` 后基本不参与真实计算。 |

注意：你当前日志中 `hand_eye_result=...hand_eye_result.json`，所以真实目标来自：

```text
base_from_hand_eye = T_base_camera @ camera_xyz
```

不是来自 `base_x_mm/scale_x` 这组模拟参数。

### 3.4 抓取姿态参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `rx_deg/ry_deg/rz_deg` | `None` | 目标 TCP 姿态。如果不传，执行时默认沿用当前 TCP 姿态。 |
| `use_current_orientation` | `False` | 显式要求使用当前 TCP 姿态。 |
| `auto_top_down_yaw` | `True` | 如果你传的是 top-down 姿态，自动保留当前 yaw 分支，减少大姿态跳变。 |
| `max_orientation_jump_deg` | `45` | 相邻 waypoint 姿态跳变超过该值则拒绝。 |

你当前 `rx_deg=None, ry_deg=None, rz_deg=None`，所以执行时会走：

```text
orientation = current_default
```

也就是当前 TCP 姿态会被复制到目标点。这样可以避免姿态突然翻转，但缺点是：当前姿态不一定适合所有目标点。尤其 `ry≈60~80°` 时，145 mm TCP offset 会把 J6 X 拉得很小，容易超限。

### 3.5 TCP offset 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `tcp_offset_x_mm` | `0` | J6 到 TCP 的 X 偏移。 |
| `tcp_offset_y_mm` | `0` | J6 到 TCP 的 Y 偏移。 |
| `tcp_offset_z_mm` | `145` | J6 到夹爪 TCP 的 Z 偏移，官方夹爪默认约 145 mm。 |

转换公式：

```text
offset_base = R(rx, ry, rz) @ [tcp_offset_x, tcp_offset_y, tcp_offset_z]
J6 = TCP - offset_base
TCP = J6 + offset_base
```

所以 TCP offset 不只是影响 Z，也会因为姿态旋转投影到 X/Y。`debug-plan` 中的 `tcp_offset_base` 就是这个投影。

### 3.6 抓取高度参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `target_z_offset_mm` | `40` | 在手眼映射得到的目标 Z 上额外加的抓取高度补偿。 |
| `approach_height_mm` | `40` | 抓取点上方 approach TCP 高度。 |
| `lift_height_mm` | `40` | 抓取后 lift TCP 高度。 |

当前规划大致是：

```text
target_tcp.z = hand_eye_z
grasp_tcp.z = target_tcp.z + target_z_offset_mm
approach_tcp.z = grasp_tcp.z + approach_height_mm
lift_tcp.z = grasp_tcp.z + lift_height_mm
```

如果物体本身很高，或者手眼 Z 偏大，再加这几项后，J6 Z 可能接近或超过可达区域。

### 3.7 放置参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `place_x_mm/place_y_mm` | `60/270` | `catch_server_1` 固定放置 TCP 基座坐标。 |
| `place_pick_x_mm/place_pick_y_mm` | `60/-270` | `catch_server_1` 从固定放置区回取时使用的 TCP 基座坐标。 |
| `place_dx_mm` | `None` | 旧逻辑：放置点相对目标 TCP 的 X 偏移；显式传 `dx/dy` 任一项时启用相对模式。 |
| `place_dy_mm` | `None` | 旧逻辑：放置点相对目标 TCP 的 Y 偏移；相对模式下未传 Y 时按 `-100` 处理。 |
| `place_z_mm` | `80` | 放置 TCP Z。 |
| `place_rx/ry/rz_deg` | `None` | 放置姿态。不传时沿用抓取姿态。 |
| `place_approach_height_mm` | `50` | 放置点上方进退高度。 |
| `stop_after_lift` | `False` | 抓取并抬升后停止，不执行放置。 |

现在默认放置点不再跟随点击目标移动。如果显式使用 `place_dx/place_dy` 旧相对模式，目标本身 Y 已经接近边界时，偏移仍可能让放置点进入另一个不可达区域。初调时可以加：

```bash
--stop-after-lift
```

先只验证抓取，不验证放置。

### 3.8 夹爪参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `open_command_mm` | `0` | 张开命令。如果已把完全张开设为零，`0` 表示张开。 |
| `grasp_command_mm` | `-100` | 闭合命令。当前项目按负值闭合。 |
| `release_command_mm` | `None` | 松开命令。不传则等于 `open_command_mm`。 |
| `effort_nm` | `1.0` | 默认夹爪力矩。 |
| `grasp_effort_nm` | `None` | 抓取闭合力矩。 |
| `release_effort_nm` | `None` | 松开力矩。 |
| `gripper_settle_s` | `0.6` | 夹爪动作后等待时间。 |
| `set_open_zero_on_start` | `False` | 启动时把当前物理位置设为夹爪零点。 |
| `zero_settle_s` | `1.5` | 设零后等待时间。 |
| `clear_gripper_error` | `False` | 抓取前清夹爪错误。 |
| `no_pre_open` | `False` | 抓取前不预张开夹爪。 |
| `no_gripper` | `False` | 只移动机械臂，不控制夹爪。 |

夹爪参数通常不会导致 `TARGET_POS_EXCEEDS_LIMIT`。超限主要来自机械臂 waypoint。

### 3.9 staged approach 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `staged_approach` | `True` | 抓取前先抬升，再在当前姿态下移动到目标上方，再切换目标姿态。 |
| `stage_height_mm` | `60` | 起始 staged lift 希望抬高的高度。 |
| `stage_max_j6_z_mm` | `450` | staged lift/XY 的保守 J6 Z 上限。 |

之前常见错误：

```text
current_j6.z = 449
stage_height = 60
stage_lift.z = 509
TARGET_POS_EXCEEDS_LIMIT
```

现在 `stage_max_j6_z_mm=450` 会阻止 staged 阶段主动生成 509 mm。但如果目标 `approach_j6.z` 本身已经过高，后续 `approach` 仍可能被 `max_j6_z_mm` 拒绝。

### 3.10 保守工作空间过滤参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `min_horizontal_radius_mm` | `0` | J6 水平半径最小值。 |
| `max_horizontal_radius_mm` | `None` | J6 水平半径最大值。不传时使用机器人配置半径。 |
| `min_j6_x_mm` | `80` | 目标类 waypoint 的 J6 X 最小值。`stage_lift` 当前点例外。 |
| `max_j6_x_mm` | `None` | J6 X 最大值。 |
| `min_j6_z_mm` | `None` | J6 Z 最小值。 |
| `max_j6_z_mm` | `500` | J6 Z 最大值。 |
| `min_target_x_mm` | `150` | 手眼映射后 target TCP X 最小值。 |
| `max_target_x_mm` | `None` | target TCP X 最大值。 |
| `min_target_y_mm` | `-250` | target TCP Y 最小值。 |
| `max_target_y_mm` | `250` | target TCP Y 最大值。 |
| `min_target_z_mm` | `None` | target TCP Z 最小值。 |
| `max_target_z_mm` | `None` | target TCP Z 最大值。 |
| `warn_margin_mm` | `25` | 距离边界小于该值时打印 warning。 |

这些限制不是 Piper 官方完整 IK 极限，而是为了初调时避免明显危险/常失败的区域。它们会提前拒绝：

- `target_tcp.x < 150`
- `target_tcp.y > 250`
- `J6.x < 80`
- `J6.z > 500`
- `stage_lift.z > 450`

如果你确认现场可达，可以逐项放宽。但不要一开始全部关掉，否则 SDK 会直接拒绝或机械臂不动。

## 4. 为什么你总是出现超限

综合你之前的日志，常见原因有四类。

### 4.1 手眼映射把物体点变到了危险区域

典型日志：

```text
base_from_hand_eye: x=139 y=285 z=89 mm
```

问题：

- `x=139` 已低于当前默认 `min_target_x=150`。
- `y=285` 已超出默认 `max_target_y=250`。
- 这说明点击点经手眼矩阵转换后，不在保守可抓区域。

可能原因：

- 手眼矩阵精度不足。
- 手眼矩阵方向用反了，例如把 `base->camera` 当成 `camera->base`。
- Windows 侧深度点不是物体中心深度，而是背景/桌面深度。
- 相机内参或分辨率不匹配，导致相机 XYZ 偏差。

### 4.2 当前姿态导致 TCP offset 把 J6 拉到基座附近

典型现象：

```text
target_tcp.x = 178
tcp_offset_base.dx ≈ 132
approach_j6.x ≈ 45
```

数学上这是对的：

```text
J6.x = TCP.x - offset_base.x
```

但对机械臂来说，`J6.x≈45` 可能已经很难或不可达。这个问题不是 TCP offset 公式错，而是“当前姿态 + 目标位置”的组合不适合。

处理方式：

- 把物体放到 `target_tcp.x` 更靠前的位置，例如 250~350 mm。
- 或测试明确 top-down 姿态，但要注意姿态跳变：

```bash
--rx-deg 180 --ry-deg 0 --rz-deg 0 --max-orientation-jump-deg 90
```

- 观察 `debug-plan` 中 `tcp_offset_base`，如果 `dx` 太大，说明当前姿态会强烈压缩 J6 X。

### 4.3 staged lift 或 approach Z 太高

典型现象：

```text
current_j6.z = 449
stage_lift.z = 509
arm_status=TARGET_POS_EXCEEDS_LIMIT
```

这不是手眼问题，也不是姿态翻转，而是 Z 高度本身被 SDK 拒绝。

当前代码已有：

```text
stage_max_j6_z_mm = 450
max_j6_z_mm = 500
```

如果仍出现 Z 超限，看 `debug-plan` 中具体是：

- `staged stage_lift` 超限：降低 `--stage-max-j6-z-mm` 或把起始姿态放低。
- `waypoint approach/lift/place_approach` 超限：降低 `target_z_offset/approach_height/lift_height/place_approach_height`，或检查手眼 Z 是否偏高。

### 4.4 放置点不可达

默认：

```text
place_x = 150
place_y = 150
place_z = 80
```

它是机械臂基座坐标系下的固定 TCP 放置点，不再基于抓取点计算。若显式传 `--place-dx-mm/--place-dy-mm`，会切换回旧的目标相对计算。初调建议：

```bash
--stop-after-lift
```

先确认抓取和抬升，再单独调放置。

## 5. 推荐排查顺序

### 第一步：只看坐标，不让机械臂动

Linux 不加 `--execute`：

```bash
python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --hand-eye-result \
  --debug-plan
```

Windows 点击或视觉检测发送点，检查：

- `camera xyz`
- `base_from_hand_eye`
- `target_tcp`
- `tcp_offset_base`
- `approach_j6/grasp_j6`

### 第二步：真实执行但不动夹爪，只抓取后停止

```bash
python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --can can0 \
  --execute \
  --hand-eye-result \
  --speed 5 \
  --debug-plan \
  --no-gripper \
  --stop-after-lift
```

### 第三步：如果 target 经常被过滤

如果报：

```text
target_tcp x=... below conservative --min-target-x-mm
target_tcp y=... exceeds conservative --max-target-y-mm
```

先不要放宽参数，优先确认：

- Windows 侧相机内参是否和当前分辨率一致。
- 检测中心点是否在物体上。
- 深度值是否是物体表面，不是背景。
- 手眼标定质量是否过差。
- 棋盘格角点参数是否正确。

### 第四步：如果 J6 x 被过滤

如果报：

```text
J6 x ... below conservative --min-j6-x-mm
```

看 `tcp_offset_base.dx`：

- 如果 `dx≈120~140`，说明当前姿态把 TCP offset 大量投影到 X。
- 这种情况下，即使 target TCP 看似可达，J6 也可能不可达。

处理：

- 把物体放到更远的 X 区域。
- 尝试 top-down 姿态。
- 或放宽 `--min-j6-x-mm` 做验证，但不要作为长期解决方案。

### 第五步：如果 Z 被过滤或 SDK 报 Z 超限

降低这些参数：

```bash
--stage-max-j6-z-mm 430
--max-j6-z-mm 470
--target-z-offset-mm 20
--approach-height-mm 30
--lift-height-mm 30
```

如果降低后还能抓到，说明之前主要是 Z 规划过高。

## 6. 一组相对保守的调试命令

只验证抓取和抬升，不控制夹爪：

```bash
python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --can can0 \
  --execute \
  --hand-eye-result \
  --speed 5 \
  --debug-plan \
  --no-gripper \
  --stop-after-lift \
  --stage-max-j6-z-mm 430 \
  --max-j6-z-mm 470 \
  --target-z-offset-mm 30 \
  --approach-height-mm 30 \
  --lift-height-mm 30
```

如果上面能稳定到达，再逐步恢复：

1. 去掉 `--stop-after-lift`，验证放置。
2. 去掉 `--no-gripper`，验证夹爪。
3. 逐步调高 `target_z_offset/approach_height/lift_height`。

## 7. 判断到底是哪一部分有问题

| 现象 | 更可能的问题 |
|---|---|
| `camera xyz` 深度明显不对 | Windows 深度采样/检测中心问题 |
| `base_from_hand_eye` 和真实物体基座坐标差很多 | 手眼矩阵或内参/分辨率问题 |
| `target_tcp` 合理，但 `J6 x` 很小 | 当前姿态 + TCP offset 投影导致 |
| `stage_lift` 第一阶段就超限 | 起始高度太高或 staged Z 规划过高 |
| `approach/grasp/lift` 超限 | 目标点、姿态、TCP offset 或高度参数组合不可达 |
| `place_approach/place` 超限 | 放置点参数不可达 |
| `ori_err` 很大 | 姿态跳变或 Euler 分支问题 |
| `pos_err` 很大且机械臂不动 | SDK 拒绝目标，通常是空间/IK/限位问题 |

## 8. 最关键的结论

你现在反复超限，通常不是单一错误，而是下面三件事叠加：

1. 手眼转换后的目标有时落在 `x` 偏小、`y` 偏大的区域。
2. 当前默认沿用当前姿态，`ry≈60~80°` 时 145 mm TCP offset 会把 J6 的 X 往回拉 100 mm 以上。
3. staged 或 approach 高度如果从较高起始位姿继续抬升，容易碰到 Piper 当前姿态下的真实可达上限。

因此排查时不要只看 `target_tcp`。必须同时看：

```text
base_from_hand_eye
target_tcp
tcp_offset_base
approach_j6 / grasp_j6 / lift_j6
staged stage_lift / stage_xy
```

只有这些都合理，真实抓取才有机会稳定成功。
