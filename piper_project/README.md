# Piper Project

Piper Project 是一个面向 Piper 机械臂的视觉抓取与传送带分拣工程。项目按运行主机拆分：

- Windows 视觉端：连接 RealSense D435i，完成 RGB/Depth 对齐、OpenCV/YOLO/YOLO-seg/GraspNet 目标定位，并通过 TCP 发送相机坐标。
- Linux/VM 控制端：连接 Piper SDK 和 CAN，完成手眼坐标变换、抓取路径规划、机械臂运动和夹爪控制。
- Web 指挥中心：用浏览器启动/停止常用 Windows 与 Linux 命令，并查看 OpenCV 预览和关键日志。

默认原则：所有会驱动真实机械臂的服务都先 dry-run，只有显式传 `--execute` 才执行实机动作。

## 项目结构

```text
piper_project/
  robot_control/
    shared/                  # 跨 Windows/Linux 的协议、路径、几何数据结构
    windows/
      vision/                # RealSense、深度、OpenCV/YOLO 检测基础模块
      client/                # Windows 视觉侧点击、YOLO、分割、OpenCV、GraspNet 客户端
      hand_eye/              # Windows 侧棋盘格采集和手眼验证
    linux/
      arm/                   # Piper SDK 机械臂控制、运动、正运动学、安全检查
      gripper/               # 官方夹爪控制、调试、诊断和配置
      catch/                 # 点击抓取、传送带抓取、固定轨迹 demo
      hand_eye/              # Linux 侧手眼采样、自动采样、求解
      tcp_calibration/       # J6 -> 夹爪 TCP 固定点标定
      server/                # 通用 TCP server
  web_command_center/        # Web 控制台
  models/                    # YOLO 权重、GraspNet 后端说明、标定数据
```

更完整的文件职责见 `PROJECT_STRUCTURE.md`。

## 数据流

```text
RealSense D435i
  -> Windows RGB/Depth 对齐
  -> OpenCV / YOLO / YOLO-seg / GraspNet 生成抓取点
  -> 相机坐标 camera XYZ
  -> TCP JSON Lines
  -> Linux catch server
  -> 手眼标定 camera -> base
  -> Piper J6 位姿序列
  -> Piper SDK / CAN
  -> 机械臂和夹爪执行
```

## 环境准备

安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

注意事项：

- `requirements.txt` 中包含 `-e ../sdk/piper_sdk`，需要保证 Piper SDK 位于项目相邻目录 `../sdk/piper_sdk`。
- Windows 视觉端需要 RealSense D435i、`pyrealsense2`、OpenCV 窗口环境，以及 `models/` 下的模型权重。
- Linux 控制端需要 CAN 设备、Piper SDK 和真实机械臂连接。
- GraspNet/AnyGrasp 不只需要 checkpoint，还需要官方后端源码和自定义算子，详见 `models/README_GRASPNET_BACKEND.md`。

Linux 配置 CAN：

```bash
./setup_can.sh can0 1000000
```

## 快速开始

### 1. 启动 Linux 传送带抓取服务

先 dry-run，不加 `--execute`，确认收到目标后的映射坐标、抓取点和路径都合理：

```bash
python3 -m robot_control.linux.catch.catch_server_1 \
  --host 0.0.0.0 \
  --port 5005 \
  --can can0 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json \
  --speed 5 \
  --debug-plan
```

确认安全后再加 `--execute`：

```bash
python3 -m robot_control.linux.catch.catch_server_1 \
  --host 0.0.0.0 \
  --port 5005 \
  --can can0 \
  --execute \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json \
  --speed 5 \
  --print-feedback
```

`catch_server_1.py` 是当前传送带抓取核心服务，支持 `out`/`in` 任务、静态选择、动态传送带时序、固定/动态 pitch、safe gripper 和 planar IK。详细调参见 `robot_control/YOLO_CONVEYOR_SORTING_ARCHITECTURE.md` 与 `robot_control/linux/catch/README.md`。

### 2. 启动 Windows 视觉客户端

OpenCV 颜色/轮廓抓取：

```bash
python -m robot_control.windows.client.opencv_grasp_send_tcp \
  --host <linux-vm-ip> \
  --port 5005 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json
```

YOLO bbox 抓取：

```bash
python -m robot_control.windows.client.yolo_grasp_send_tcp \
  --host <linux-vm-ip> \
  --port 5005
```

YOLO segmentation 抓取：

```bash
python -m robot_control.windows.client.yolo_seg_grasp_send_tcp \
  --host <linux-vm-ip> \
  --port 5005 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json
```

这些客户端默认是静态模式，会在终端打印按深度排序的目标。输入 `out 1`、`in 2` 这类命令发送目标；按 `q` 或 `Esc` 退出。动态传送带模式使用 `dynamic` 子模式，并需要配置传送带速度和抓取窗口，例如：

```bash
python -m robot_control.windows.client.opencv_grasp_send_tcp dynamic \
  --host <linux-vm-ip> \
  --port 5005 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json \
  --conveyor-speed-mm-s 80 \
  --dynamic-pick-x-mm 350 \
  --dynamic-pick-y-mm -30
```

### 3. 使用 Web 指挥中心

```bash
python3 -m web_command_center.server --host 127.0.0.1 --port 8088
```

打开：

```text
http://127.0.0.1:8088
```

建议把 Web 服务运行在 Windows 视觉端。页面会执行输入框里的命令，默认只监听 `127.0.0.1`；需要局域网访问时再改成 `--host 0.0.0.0`。详细说明见 `web_command_center/README.md`。

## 常用命令

### 机械臂 CLI

```bash
python3 -m robot_control.linux.arm.arm_cli status
python3 -m robot_control.linux.arm.arm_cli enable
python3 -m robot_control.linux.arm.arm_cli resume
python3 -m robot_control.linux.arm.arm_cli pose 150 0 215 0 85 0 --speed 20 --mode P
python3 -m robot_control.linux.arm.arm_cli joints 0 1.5 -1 0 0.5 -1.25 --speed 10
```

`arm_cli joints` 校验目标是否在机械臂的关节范围内，然后直接下发关节运动命令。
`arm_cli resume` 只发送软件急停解除指令，不会下发运动指令。

### 夹爪调试

```bash
python3 -m robot_control.linux.gripper.gripper_debug status --can can0 --enable --pretty
python3 -m robot_control.linux.gripper.gripper_debug command open --can can0 --width 50 --effort 1.0
python3 -m robot_control.linux.gripper.diagnose_j6_gripper --can can0
```

设置完全张开位置为夹爪零点：

```bash
python3 -m robot_control.linux.catch.set_gripper_zero --can can0 --yes --verify
```

### 点击抓取

Linux 侧：

```bash
python3 -m robot_control.linux.catch.catch_server \
  --host 0.0.0.0 \
  --port 5005 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json \
  --speed 5
```

Windows 侧：

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

确认目标点和路径后，Linux 侧再加 `--execute`。

### 固定轨迹实机链路验证

不依赖相机，适合先验证机械臂和夹爪完整动作链路：

```bash
python3 -m robot_control.linux.catch.scripted_catch_demo \
  --can can0 \
  --speed 5 \
  --no-gripper \
  --print-feedback
```

默认不加 `--execute` 时只打印路径和做边界检查。确认安全后再逐步加入 `--execute` 和夹爪动作。

## 标定流程

### 手眼标定 camera -> base

Linux 启动采样服务器：

```bash
python3 -m robot_control.linux.hand_eye.collect_server \
  --host 0.0.0.0 \
  --port 5010 \
  --can can0 \
  --mode eye_to_hand \
  --overwrite
```

Windows 采集棋盘格：

```bash
python -m robot_control.windows.hand_eye.capture_chessboard \
  --host <linux-vm-ip> \
  --port 5010 \
  --corners-rows 8 \
  --corners-cols 11 \
  --checker-size-mm 15
```

采集 20-30 组后在 Linux 求解：

```bash
python3 -m robot_control.linux.hand_eye.solve_hand_eye \
  --samples robot_control/linux/hand_eye/hand_eye_samples.json \
  --output robot_control/linux/hand_eye/hand_eye_result.json \
  --mode eye_to_hand \
  --method tsai
```

详细采样建议、自动采集和验证方法见 `robot_control/linux/hand_eye/README.md`。

### 夹爪 TCP 标定 J6 -> gripper_tcp

```bash
python3 -m robot_control.linux.tcp_calibration.collect_tcp_data --can can0 --overwrite
python3 -m robot_control.linux.tcp_calibration.solve_tcp
python3 -m robot_control.linux.tcp_calibration.validate_tcp --collect --can can0 --overwrite
```

该流程标定的是 `J6 -> gripper_tcp` 平移偏移，不是相机外参或手眼矩阵。详细说明见 `robot_control/linux/tcp_calibration/README.md`。

## 测试

运行不依赖真实硬件的单元测试：

```bash
python3 -m unittest discover -s robot_control -p '*test.py'
```

如果要连同根目录兼容入口和后端测试一起发现：

```bash
python3 -m unittest discover -s . -p '*test.py'
```

更多手动脚本参考见 `robot_control/TESTING.md`。

## 安全检查

实机运行前至少确认：

- CAN 接口正确，现场没有其他进程占用 `can0`。
- 急停、限位、工作空间、放置点和传送带区域都符合当前安装环境。
- 第一次运行使用低速、小位移、空场景和 dry-run。
- `hand_eye_result.json` 的方向必须是 `camera -> base`，否则抓取点会映射错误。
- 夹爪零点、张开/闭合命令和力矩参数必须和当前物理安装一致。
- 只有在确认路径、安全边界和目标映射都正确后，才给 Linux server 添加 `--execute`。
