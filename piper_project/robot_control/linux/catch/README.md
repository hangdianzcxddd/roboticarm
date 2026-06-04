# 点击抓取 Server

这个目录提供 Linux 侧点击抓取入口。它复用现有 Windows 客户端：

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

Linux 侧收到相机 XYZ 后，可以使用手眼标定结果
`robot_control/linux/hand_eye/hand_eye_result.json` 中的 `camera -> base`
变换，把点击点转换成一个“目标夹爪 TCP 位姿”。如果没有传
`--hand-eye-result`，仍会沿用旧的临时小范围映射。随后使用官方夹爪偏移：

```text
J6 -> gripper TCP: x=0 mm, y=0 mm, z=145 mm
```

反算需要下发给机械臂的 J6 位姿，并执行：

```text
张开夹爪 -> 移动到目标上方 -> 下降到目标 -> 闭合夹爪 -> 抬升 -> 移动到放置点 -> 松开 -> 撤离
```

## 重要边界

- `145 mm` 只补偿官方夹爪末端相对 J6 的偏移。
- 加 `--hand-eye-result` 后，点击点会使用手眼标定 `camera -> base` 变换。
- 不加 `--hand-eye-result` 时，仍使用 `shared/simulated_mapping.py` 的临时映射。
- `--use-current-pose-as-base` 只适用于临时映射，不能和 `--hand-eye-result` 同时使用。
- 默认假设你已经把“完全张开 100 mm”的物理位置设为了夹爪零点；此时 `0 mm` 命令表示完全张开，`-100 mm` 命令表示向闭合方向运动。
- 执行时如果没有传 `--rx-deg/--ry-deg/--rz-deg`，目标 TCP 姿态默认沿用当前反馈姿态，避免大角度翻转。
- 默认放置 TCP 点为点击目标的 `x + 0 mm, y - 100 mm, z=80 mm`。若现场坐标方向相反，改 `--place-dy-mm` 或显式传 `--place-x-mm/--place-y-mm/--place-z-mm`。
- 第一次执行前请用很低速度、小位移、空场景测试。

## Linux 侧命令：使用手眼标定结果

第一次建议先不加 `--execute`，只 dry-run，点击几个点看 Linux 打印的
`target_tcp` 是否落在真实物体附近且在安全工作空间内：

```bash
cd /home/zcxddd/projects/roboticarm/piper_project

python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json \
  --speed 5 \
  --command-duration-s 1.0 \
  --command-period-s 0.02 \
  --print-feedback \
  --target-z-offset-mm 40 \
  --approach-height-mm 40 \
  --lift-height-mm 40 \
  --open-command-mm 0 \
  --grasp-command-mm -100 \
  --release-command-mm 0 \
  --grasp-effort-nm 1.5
```

确认目标点正确后，再加 `--execute`。初次实抓建议保留
`--target-z-offset-mm 40`，让夹爪先抓目标上方 40 mm 附近，确认 XY 方向正确后
逐步调小：

```bash
python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --can can0 \
  --execute \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json \
  --speed 5 \
  --command-duration-s 1.5 \
  --command-period-s 0.02 \
  --print-feedback \
  --target-z-offset-mm 40 \
  --approach-height-mm 40 \
  --lift-height-mm 40 \
  --open-command-mm 0 \
  --grasp-command-mm -100 \
  --release-command-mm 0 \
  --grasp-effort-nm 1.5
```

## Linux 侧命令：旧临时映射

从项目根目录运行：

```bash
cd /home/zcxddd/projects/roboticarm/piper_project

python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --execute \
  --speed 5 \
  --use-current-pose-as-base \
  --max-delta-x-mm 20 \
  --max-delta-y-mm 20 \
  --max-delta-z-mm 10 \
  --command-duration-s 1.5 \
  --command-period-s 0.02 \
  --print-feedback \
  --open-command-mm 0 \
  --grasp-command-mm -100 \
  --release-command-mm 0 \
  --grasp-effort-nm 1.5
```

## 单独设置夹爪零点

先手动让夹爪完全张开到 100 mm，然后运行：

```bash
python3 -m robot_control.linux.catch.set_gripper_zero --can can0 --yes --verify
```

设置完成后，`0 mm` 命令表示完全张开，抓取闭合可以使用 `--grasp-command-mm -100`。

如果仍要让抓取 server 在启动时设置当前夹爪物理位置为零点，也可以加：

```bash
--set-open-zero-on-start
```

## 不依赖相机的完整流程测试

如果手眼标定或点击目标还不稳定，先用固定脚本验证完整动作链路：

```bash
python3 -m robot_control.linux.catch.scripted_catch_demo \
  --can can0 \
  --speed 5 \
  --approach-height-mm 50 \
  --descend-mm 15 \
  --lift-height-mm 60 \
  --place-dy-mm 40 \
  --place-z-mm 80 \
  --stage-height-mm 60 \
  --continue-on-waypoint-error \
  --no-gripper \
  --print-feedback
```

不加 `--execute` 时只连接读取当前位姿，打印抓取 TCP、放置 TCP 和所有目标 J6，并在发送命令前做
XYZ、姿态、水平半径和可选保守高度/半径边界检查。确认目标都在安全范围内后再加：

```bash
--execute
```

默认抓取 TCP 使用固定安全测试点 `x=300 mm, y=0 mm, z=80 mm`，姿态默认保持当前 J6 的
`rx/ry/rz`；随后执行 approach、下降、闭合、抬升、移动到附近放置点、松开、撤离、恢复起始姿态并回到起始
J6。这个脚本不使用 Windows 点击、相机深度或 `hand_eye_result.json`，适合先验证“机械臂能否完整跑完路径”。
路径确认后去掉 `--no-gripper`，再验证夹爪闭合和松开。不要一开始指定很远的 `--place-x-mm/--place-y-mm`，
先用默认 `--place-dy-mm 40` 在当前位置附近完成流程。

如果你已经量到一组明确可达的基座坐标，可以直接输入抓取/放置 TCP。建议仍然先 dry-run：

```bash
python3 -m robot_control.linux.catch.scripted_catch_demo \
  --can can0 \
  --speed 5 \
  --target-x-mm 180 \
  --target-y-mm 0 \
  --target-z-mm 40 \
  --place-x-mm 180 \
  --place-y-mm 40 \
  --place-z-mm 80 \
  --approach-height-mm 50 \
  --lift-height-mm 60 \
  --no-gripper \
  --print-feedback
```

若现场某些区域虽然在官方半径内但 IK 经常失败，可以加更保守的边界，例如：

```bash
--min-horizontal-radius-mm 80 --max-horizontal-radius-mm 500 --min-j6-z-mm 80 --max-j6-z-mm 500
```

`catch_server` 默认已经启用一组保守抓取边界：`--stage-max-j6-z-mm 450`、
`--max-j6-z-mm 500`、`--min-j6-x-mm 80`、`--min-target-x-mm 150`、
`--min-target-y-mm -250`、`--max-target-y-mm 250`。这些边界用于初调阶段提前拒绝
`J6 x≈0~50 mm`、`target y≈300 mm` 或 `stage_lift z>450 mm` 这类 Piper SDK
经常直接报 `TARGET_POS_EXCEEDS_LIMIT` 的目标。确认现场确实可达后再逐项放宽。

如果某个 waypoint 已经接近但仍因为 8mm/8deg 阈值失败，`--continue-on-waypoint-error` 会打印警告并继续，
用于先验证完整流程。等流程能跑完后再去掉这个参数，逐步收紧精度。

如果启动时报 `Piper arm enable timed out`，先确认没有另一个 `catch_server` 或 demo 脚本还连着 `can0`。
若机械臂本来已经处于使能状态，可以在 demo 命令中临时加 `--no-enable`，让脚本只连接反馈和下发命令，
不再调用 `EnablePiper()`。如果机械臂未使能，`--no-enable` 不会让它运动。

## 常用调参

- `--hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json`：使用手眼标定 `camera -> base` 变换。
- `--tcp-offset-z-mm 145`：官方夹爪默认值，等于 `0.145 m`。
- `--approach-height-mm 40`：先移动到目标 TCP 上方多少毫米。
- `--lift-height-mm 40`：抓取后抬升多少毫米。
- 不传 `--rx-deg/--ry-deg/--rz-deg`：执行时默认沿用当前 TCP 姿态，适合手眼抓取初调，避免机械臂姿态翻转过大。
- `--rx-deg 180 --ry-deg 0 --rz-deg 0`：显式 Top-Down 抓取姿态，工具 Z 轴指向机器人基座 `-Z`，即夹爪垂直向下。执行时默认会自动保留当前 `rz` 分支，例如当前反馈 `rz=180` 时会使用等价的 `rx=180 ry=0 rz=180`。
- `--no-auto-top-down-yaw`：关闭上面的自动 `rz` 分支选择，严格使用命令行给出的 `--rx-deg/--ry-deg/--rz-deg`。
- `--max-orientation-jump-deg 45`：拒绝相邻 J6 waypoint 之间超过 45 度的姿态跳变；显式 top-down 调试时可临时放宽。
- `--stage-height-mm 60`：执行抓取前先用当前姿态抬高并移动到目标上方，再切换到抓取姿态。默认开启；如果要恢复直接去 approach，可加 `--no-staged-approach`。
- `--stage-max-j6-z-mm 450`：限制 staged approach 的预抬升高度，避免从当前 `z≈449` 继续抬到 `z≈509` 后被 SDK 拒绝。
- `--min-j6-x-mm 80`：拒绝目标 waypoint 中 J6 太靠近基座的规划。当前姿态 `ry≈66°` 时，145mm TCP 偏移会把 TCP 的 X 往回拉约 130mm，容易生成 `J6 x≈0~50 mm`。
- `--min-target-x-mm/--min-target-y-mm/--max-target-y-mm`：对手眼映射后的目标 TCP 做保守过滤；如果点击明显可达物体却被这些条件拦住，优先检查手眼矩阵和相机深度。
- `--target-z-offset-mm 40`：对点击点做额外 Z 修正。值越小，抓取 TCP 越接近点击点。
- `--open-command-mm 0`：张开命令。完全张开设零后，`0 mm` 表示张开。
- `--grasp-command-mm -100`：抓取闭合命令。完全张开设零后，`-100 mm` 表示向闭合方向走 100 mm。
- `--release-command-mm 0`：放置时松开命令。
- `--effort-nm 1.0`：默认夹爪力矩。
- `--grasp-effort-nm 1.5`：抓取闭合力矩，可单独覆盖。
- `--release-effort-nm 0.4`：松开力矩，可单独覆盖。
- `--place-dx-mm 0 --place-dy-mm -100 --place-z-mm 80`：默认相对目标的放置 TCP 点；也可用 `--place-x-mm/--place-y-mm` 固定到绝对基座坐标。
- `--place-approach-height-mm 50`：放置点上方的进退高度。
- `--clear-gripper-error`：抓取前先发夹爪清错。
- `--no-gripper`：只移动机械臂，不执行夹爪张合。

## 日志怎么看

启动时会先打印基础状态：

```text
Using hand-eye camera->base transform: robot_control/linux/hand_eye/hand_eye_result.json
Connecting and enabling Piper arm...
Piper arm enabled
Listening on 0.0.0.0:5005, execute=True, speed=5, tcp_offset_j6_mm=(0.0, 0.0, 145.0)
Catch args: Namespace(...)
```

- `Using hand-eye...`：当前使用手眼标定结果，把 Windows 发来的相机坐标转换到机器人基座坐标。
- `Piper arm enabled`：SDK 使能成功，只说明可以发命令，不代表每个目标都可达。
- `execute=True`：会真实运动；如果是 `False`，只计算目标不运动。
- `speed=5`：运动速度百分比，初调建议保持低速。
- `tcp_offset_j6_mm=(0, 0, 145)`：J6 到夹爪 TCP 的偏移，单位 mm。
- `Catch args`：本次启动的完整参数。调试时先看这里确认参数有没有按预期生效。

收到 Windows 点击后，会打印连接和当前 J6 反馈：

```text
Client connected: ('192.168.28.1', 58801)
Before catch J6: x=56.127 y=0.000 z=213.266 rx=0.000 ry=84.999 rz=0.000
```

- `Client connected`：Windows 客户端已经连上并发来了点击目标。
- `Before catch J6`：抓取前机械臂反馈的 J6 位姿，位置单位 mm，姿态单位 deg。
- `x/y/z`：J6 在机器人基座坐标系下的位置，不是相机坐标，也不是 TCP 坐标。
- `rx/ry/rz`：J6 姿态。它会用于判断 waypoint 是否到达。

如果 waypoint 没到，会打印错误：

```text
Error: waypoint approach not reached:
target=x=56.298 y=216.728 z=231.682 rx=0.000 ry=84.999 rz=0.000
last=x=56.127 y=0.000 z=213.266 rx=0.000 ry=84.999 rz=0.000
pos_err=217.509 mm ori_err=0.000 deg
```

- `waypoint approach`：当前失败在“移动到目标上方”这一步，还没开始下降抓取。
- `target`：本次下发给机械臂的目标 J6 位姿。
- `last`：等待超时前最后一次读到的 J6 反馈。
- `pos_err`：`target` 和 `last` 的位置距离。这个值小于 `--position-tolerance-mm` 才认为位置到达。
- `ori_err`：`target` 和 `last` 的旋转角误差。这个值小于 `--orientation-tolerance-deg` 才认为姿态到达。

上面这个例子里，`ori_err=0`，说明姿态没有问题；`pos_err=217.509 mm`，而且 `last` 和
`Before catch J6` 完全一样，说明机械臂没有朝第一个目标移动。此时优先排查运动控制状态、
目标是否可达、Y 方向是否超出当前姿态可达范围，而不是排查 Windows 是否连接成功。

注意：执行时如果没有传 `--rx-deg/--ry-deg/--rz-deg`，抓取目标会默认使用当前反馈 TCP 姿态，
日志中会显示 `orientation=current_default`。如果要测试 Top-Down 垂直向下抓取，显式传
`--rx-deg 180 --ry-deg 0 --rz-deg 0`；若姿态跳变超过默认 45 度，需要确认安全后再放宽
`--max-orientation-jump-deg`。

默认 Top-Down 执行还会打印 `orientation=auto_top_down_yaw`。这表示目标仍然是垂直向下，但 `rz`
会贴近当前反馈分支。以你的日志 `rx=180 ry=66 rz=180` 为例，旧目标 `rx=180 ry=0 rz=0`
需要 180 度姿态跳变；自动分支会改成 `rx=180 ry=0 rz=180`，更容易被 Piper 接受。

## 按现象调试

- `Client connected` 没出现：先排查 Windows 到 Linux 的 IP、端口、防火墙。
- `target` 明显离谱，比如几千 mm 或负高度：优先排查手眼标定、深度值、m/mm 单位。
- `ori_err` 很大，`pos_err` 也不变：目标姿态可能不可达，先不传 `--rx-deg/--ry-deg/--rz-deg`，用当前姿态做小范围运动测试。
- `ori_err=0` 但 `pos_err` 很大，`last` 不变：姿态没问题，但机械臂没有移动到目标位置，优先看机械臂状态、急停/示教/报错、目标位置可达性。
- `pos_err` 逐渐变小但超时：可以适当增大 `--motion-timeout-s` 或降低目标距离，确认速度和轨迹是否合理。
- 只想验证目标计算，不想运动：去掉 `--execute`，点击后看 `target_tcp` 和 `grasp_j6` 是否合理。

定位手眼映射问题时，Linux server 启动加 `--debug-plan`。每次点击会在运动前打印：

```text
camera xyz -> base_from_hand_eye -> target_tcp/grasp_tcp/place_tcp -> each J6 waypoint
```

如果不想连接机械臂，可以把 Windows 侧打印的 `X/Y/Z` 复制到离线脚本：

```bash
python3 -m robot_control.linux.catch.debug_hand_eye_point \
  --x-m 0.10 \
  --y-m -0.05 \
  --z-m 0.30 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json
```

当前 `hand_eye_result.json` 启动时也会打印质量摘要。如果看到 `rotation RMS is high`
或 `translation RMS is high`，优先重新采集/求解手眼标定。

## Windows 侧命令

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

点击 RGB 图像后，Windows 会发送相机 XYZ；Linux 会打印目标 TCP、反算后的 J6 抓取位姿和执行状态。

## Windows 侧 OpenCV 自动视觉抓取

先在 Linux 侧启动带手眼矩阵的抓取服务：

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

Windows 侧可以用 OpenCV 背景减除检测物体框，并把检测中心的相机 XYZ 按和点击脚本相同的协议发送给
`catch_server`：

```bash
python -m robot_control.windows.client.opencv_grasp_send_tcp \
  --host <linux-vm-ip> \
  --port 5005 \
  --algorithm background \
  --target-policy largest
```

默认流程会先学习背景，物体不要放进画面；学习结束后把物体放入画面。检测稳定后按 `g` 发送当前目标。
如果想让脚本稳定后自动发送，加 `--auto-send`；初调时建议先用 `--dry-run` 或先不加 `--auto-send`，
确认 Linux `--debug-plan` 中的 `base_from_hand_eye`、`target_tcp`、`tcp_offset_base` 和各 J6 waypoint 合理。

如果物体有明显颜色，也可以用 HSV 阈值：

```bash
python -m robot_control.windows.client.opencv_grasp_send_tcp \
  --host <linux-vm-ip> \
  --algorithm color \
  --lower-hsv 35 60 60 \
  --upper-hsv 85 255 255
```
