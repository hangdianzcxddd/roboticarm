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
- 默认放置 TCP 点为 `x=0 mm, y=150 mm, z=150 mm`，即机器人基座左侧 15 cm、高 15 cm。若现场坐标方向相反，把 `--place-y-mm` 改成 `-150`。
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
  --rx-deg 0 \
  --ry-deg 85 \
  --rz-deg 0 \
  --target-z-offset-mm 20 \
  --approach-height-mm 60 \
  --lift-height-mm 60 \
  --open-command-mm 0 \
  --grasp-command-mm -100 \
  --release-command-mm 0 \
  --grasp-effort-nm 1.5 \
  --place-x-mm 0 \
  --place-y-mm 150 \
  --place-z-mm 150
```

确认目标点正确后，再加 `--execute`。初次实抓建议保留
`--target-z-offset-mm 20`，让夹爪先抓目标上方 20 mm 附近，确认 XY 方向正确后
逐步调到 `0` 或负值：

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
  --rx-deg 0 \
  --ry-deg 85 \
  --rz-deg 0 \
  --target-z-offset-mm 20 \
  --approach-height-mm 60 \
  --lift-height-mm 60 \
  --open-command-mm 0 \
  --grasp-command-mm -100 \
  --release-command-mm 0 \
  --grasp-effort-nm 1.5 \
  --place-x-mm 0 \
  --place-y-mm 150 \
  --place-z-mm 150
```

## Linux 侧命令：旧临时映射

从项目根目录运行：

```bash
cd /home/zcxddd/projects/roboticarm/piper_project

python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --execute \
  --speed 10 \
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
  --grasp-effort-nm 1.5 \
  --place-x-mm 0 \
  --place-y-mm 150 \
  --place-z-mm 150
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

## 常用调参

- `--hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json`：使用手眼标定 `camera -> base` 变换。
- `--tcp-offset-z-mm 145`：官方夹爪默认值，等于 `0.145 m`。
- `--approach-height-mm 50`：先移动到目标 TCP 上方多少毫米。
- `--lift-height-mm 50`：抓取后抬升多少毫米。
- `--target-z-offset-mm 0`：对点击点做额外 Z 修正。负数表示更低。
- `--open-command-mm 0`：张开命令。完全张开设零后，`0 mm` 表示张开。
- `--grasp-command-mm -100`：抓取闭合命令。完全张开设零后，`-100 mm` 表示向闭合方向走 100 mm。
- `--release-command-mm 0`：放置时松开命令。
- `--effort-nm 1.0`：默认夹爪力矩。
- `--grasp-effort-nm 1.5`：抓取闭合力矩，可单独覆盖。
- `--release-effort-nm 0.4`：松开力矩，可单独覆盖。
- `--place-x-mm 0 --place-y-mm 150 --place-z-mm 150`：放置 TCP 点。
- `--place-approach-height-mm 50`：放置点上方的进退高度。
- `--clear-gripper-error`：抓取前先发夹爪清错。
- `--no-gripper`：只移动机械臂，不执行夹爪张合。

## Windows 侧命令

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

点击 RGB 图像后，Windows 会发送相机 XYZ；Linux 会打印目标 TCP、反算后的 J6 抓取位姿和执行状态。
