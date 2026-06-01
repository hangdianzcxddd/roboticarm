# Piper Project

## 架构

工程现在按运行主机拆分：

```text
robot_control/
  shared/                  # Windows 和 Linux 共用的数据结构、协议、路径
  windows/
    vision/                # D435i、深度、OpenCV/YOLO 检测
    client/                # Windows 视觉侧手动/联调入口
    task/                  # 视觉侧抓取任务生成
  linux/
    arm/                   # Piper SDK 机械臂控制、运动、安全、测试
    gripper/               # 夹爪控制、调试、配置、测试
    server/                # Linux VM TCP server
    tcp_calibration/       # TCP 标定采集、求解、验证入口
    config/                # Linux 机器人配置
```

旧的分散目录已经删除，后续代码和测试都放到 `robot_control/` 对应模块内。根目录的 `main.py` 和 `gripper_debug.py` 仅作为临时兼容入口保留。

## 数据流

```text
Windows 主机
  D435i RGB+Depth -> Vision Pipeline -> TCP JSON task

Linux VM
  TCP Server -> 任务/轨迹 -> Piper SDK -> CAN -> Piper 机械臂/夹爪
```

## Linux 侧命令

机械臂 CLI：

```bash
python3 -m robot_control.linux.arm.arm_cli status
python3 -m robot_control.linux.arm.arm_cli enable
python3 -m robot_control.linux.arm.arm_cli pose 150 0 215 0 85 0 --speed 20 --mode P
```

夹爪调试：

```bash
python3 -m robot_control.linux.gripper.gripper_debug status --can can0 --enable --pretty
python3 -m robot_control.linux.gripper.gripper_debug command open --can can0 --width 50 --effort 1.0
```

J6 驱动板 / 末端转接板 / 夹爪链路诊断：

```bash
python3 -m robot_control.linux.gripper.diagnose_j6_gripper --can can0
python3 -m robot_control.linux.gripper.diagnose_j6_gripper --can can0 --json
```

Linux VM TCP server：

```bash
python3 -m robot_control.linux.server.tcp_server --host 0.0.0.0 --port 5005
```

实机执行前再加 `--execute`，并先确认机械臂工作空间安全。

## Windows 侧命令

点击 RGB 图发送相机坐标到 Linux VM：

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

单机视觉调试：

```bash
python -m robot_control.windows.client.click_to_camera_xyz
python -m robot_control.windows.client.opencv_detection_to_camera_xyz
python -m robot_control.windows.client.yolo_detection_to_camera_xyz
```

## 测试

不连接真实硬件的测试：

```bash
python3 -m unittest discover -s . -p '*test.py'
```

后续新增测试应和模块放在一起，例如：

```text
robot_control/linux/gripper/gripper_test.py
robot_control/windows/vision/detector_test.py
```

## 安全

真实机械臂运行前确认 CAN 口、工作空间、急停、速度、关节限位、停放姿态都符合当前安装环境。默认 TCP server 是 dry-run，不会移动机械臂。
