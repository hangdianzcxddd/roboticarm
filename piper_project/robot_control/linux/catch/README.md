# 点击抓取 Server

这个目录提供 Linux 侧点击抓取入口。它复用现有 Windows 客户端：

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

Linux 侧收到相机 XYZ 后，会沿用当前项目的临时小范围映射，把点击点转换成一个“目标夹爪 TCP 位姿”。随后使用官方夹爪偏移：

```text
J6 -> gripper TCP: x=0 mm, y=0 mm, z=145 mm
```

反算需要下发给机械臂的 J6 位姿，并执行：

```text
打开夹爪 -> 移动到目标上方 -> 下降到目标 -> 闭合夹爪 -> 抬升
```

## 重要边界

- `145 mm` 只补偿官方夹爪末端相对 J6 的偏移。
- 当前相机点到机器人坐标仍然是 `shared/simulated_mapping.py` 的临时映射，不是手眼标定。
- 使用 `--use-current-pose-as-base` 时，基准点是当前夹爪 TCP，不是当前 J6。
- 第一次执行前请用很低速度、小位移、空场景测试。

## Linux 侧命令

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
  --print-feedback
```

## 常用调参

- `--tcp-offset-z-mm 145`：官方夹爪默认值，等于 `0.145 m`。
- `--approach-height-mm 50`：先移动到目标 TCP 上方多少毫米。
- `--lift-height-mm 50`：抓取后抬升多少毫米。
- `--target-z-offset-mm 0`：对点击点做额外 Z 修正。负数表示更低。
- `--open-width-mm 70`：抓取前打开宽度。
- `--close-width-mm 0`：抓取闭合目标宽度。
- `--effort-nm 1.0`：夹爪力矩。
- `--clear-gripper-error`：抓取前先发夹爪清错。
- `--no-gripper`：只移动机械臂，不执行夹爪张合。

## Windows 侧命令

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

点击 RGB 图像后，Windows 会发送相机 XYZ；Linux 会打印目标 TCP、反算后的 J6 抓取位姿和执行状态。
