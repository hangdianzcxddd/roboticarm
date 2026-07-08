# Piper Project 工程结构介绍

本文档说明 `piper_project` 当前工程的目录分层、每个子目录职责，以及主要文件的用途。
范围以项目源码、配置、模型、标定数据和文档为主；`.git`、`__pycache__`、`.pytest_cache`
等运行/工具生成目录单独说明，不作为业务源码维护。

## 1. 总体定位

该工程是一个 Piper 机械臂视觉抓取与传送带分拣项目，按运行主机拆分为：

- `robot_control/windows/`：Windows 视觉端，负责 RealSense D435i 采集、OpenCV/YOLO 检测、深度反投影、手眼点击验证和 TCP 发送。
- `robot_control/linux/`：Linux/VM 控制端，负责 Piper SDK/CAN 连接、机械臂运动、夹爪控制、抓取 server、手眼标定和 TCP 标定。
- `robot_control/shared/`：Windows 与 Linux 共用的数据结构、路径、坐标和 JSON Lines 通信协议。
- `models/`：YOLO 模型权重与 Ultralytics 本地运行配置。

典型数据流为：

```text
RealSense D435i
  -> Windows RGB/Depth 对齐
  -> OpenCV/YOLO 目标检测
  -> 像素 + 深度反投影为 camera XYZ
  -> TCP JSON Lines 发送给 Linux
  -> Linux 通过手眼结果映射到 robot base
  -> 抓取动作规划
  -> Piper SDK / CAN 控制机械臂和夹爪
```

## 2. 根目录文件

| 文件/目录 | 作用 |
| --- | --- |
| `README.md` | 项目总览，说明跨平台架构、数据流、常用运行命令和安全提示。 |
| `PROJECT_STRUCTURE.md` | 本文件，说明工程目录和文件职责。 |
| `requirements.txt` | Python 依赖列表，包括 `numpy`、`opencv-python`、`pyrealsense2`、`ultralytics`、`python-can`、`pytest` 和本地 Piper SDK editable 依赖。 |
| `main.py` | 兼容入口，转发/导出 Linux 侧 Piper 机械臂控制相关能力，保留给旧调用方式使用。 |
| `gripper_debug.py` | 兼容入口，转发/导出 Linux 侧 Piper 夹爪调试工具，保留给旧调用方式使用。 |
| `__init__.py` | 将项目根目录作为 Python 包的一部分；当前基本为空。 |
| `belt_depth_plane.json` | 传送带空带深度平面标定结果，供视觉端做深度平面补偿或空带参考。 |
| `librealsense.pgp` | Intel RealSense / librealsense 软件源相关 PGP key 文件。 |
| `6.24阶段性总结` | 2026-06-24 阶段性总结文档，记录传送带快递包裹视觉分拣任务当时的完成情况、问题和后续方向。 |

## 3. `models/`

模型和 Ultralytics 运行配置目录。

| 文件/目录 | 作用 |
| --- | --- |
| `models/yolo26n.pt` | 默认 YOLO 检测模型权重，供 `yolo_grasp_send_tcp.py`、`yolo_detection_to_camera_xyz.py` 等检测脚本加载。 |
| `models/yolo26n-seg.pt` | YOLO segmentation 模型权重，供分割抓取客户端使用。 |
| `models/yolo26n-seg-best.pt` | 训练/调优后的 YOLO segmentation best 权重，通常用于更贴近当前场景的分割抓取。 |
| `models/Ultralytics/settings.json` | Ultralytics 本地配置文件，记录模型目录、数据目录等运行设置。 |
| `models/Ultralytics/persistent_cache.json` | Ultralytics 持久缓存文件，记录运行时缓存状态。 |

## 4. `robot_control/`

项目主包，承载跨平台控制、视觉、协议和测试代码。

| 文件/目录 | 作用 |
| --- | --- |
| `robot_control/__init__.py` | Python 包标识文件。 |
| `robot_control/TESTING.md` | 测试和手动脚本参考，列出无硬件测试命令、模型下载、RealSense 调试、TCP 联调和手眼标定命令。 |
| `robot_control/YOLO_CONVEYOR_SORTING_ARCHITECTURE.md` | 机械臂传送带视觉分拣项目技术架构文档，重点描述 Windows YOLO 客户端与 Linux `catch_server_1.py` 的端到端流程。 |
| `robot_control/shared/` | Windows/Linux 共用协议、路径、几何数据结构和临时坐标映射。 |
| `robot_control/windows/` | Windows 视觉端代码。 |
| `robot_control/linux/` | Linux 控制端代码。 |

## 5. `robot_control/shared/`

跨平台共享模块。这里的代码不应依赖某一侧硬件环境，主要用于消息协议、路径和基础数据结构。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `geometry.py` | 定义共享几何数据结构，例如 `EndPose`；包含 SDK 姿态单位换算常量和默认 top-down TCP 姿态。 |
| `paths.py` | 定义项目根路径、模型目录等共享路径常量。 |
| `protocol.py` | Windows 视觉端向 Linux 运动端发送 `camera_point` 的 JSON Lines 协议；包含 `CameraPointCommand`、ACK/ERROR 消息、编码/解码和 socket 发送函数。 |
| `protocol_test.py` | `protocol.py` 的单元测试，验证消息编码、解码和相机点协议行为。 |
| `hand_eye_protocol.py` | 手眼标定人工采样协议；Windows 发送棋盘格 target-to-camera 位姿，Linux 返回采样 ACK/ERROR。 |
| `hand_eye_protocol_test.py` | `hand_eye_protocol.py` 的单元测试。 |
| `hand_eye_auto_protocol.py` | 自动手眼采样协议；Linux 请求 Windows 视觉服务检测棋盘格，Windows 返回检测位姿和质量信息。 |
| `simulated_mapping.py` | 没有真实手眼标定时使用的临时 camera-to-robot 映射，主要用于早期点击抓取联调。 |

## 6. `robot_control/linux/`

Linux/VM 机械臂控制端。该层通常需要 Piper SDK、CAN 设备和现场安全确认。

| 文件/目录 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `arm/` | Piper 机械臂控制、运动、正运动学和安全检查。 |
| `catch/` | 点击抓取、YOLO 抓取、连续抓取和抓取流程调试服务。 |
| `config/` | Linux 侧机器人配置。 |
| `gripper/` | Piper 夹爪控制、调试、诊断和配置。 |
| `hand_eye/` | Linux 侧手眼标定采样、自动采样和求解工具。 |
| `server/` | 早期/通用 TCP server，接收相机点并映射为运动命令。 |
| `tcp_calibration/` | 夹爪 TCP 固定点多姿态标定工具。 |

### 6.1 `robot_control/linux/arm/`

Piper 机械臂底层控制封装和运动基础能力。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `piper_arm.py` | Piper SDK V2 的薄封装，提供单位安全的机械臂连接、使能、位姿/关节命令和反馈读取接口。 |
| `piper_arm_test.py` | `PiperController` 相关行为的单元测试，使用 fake SDK 验证运动命令转换。 |
| `arm_cli.py` | Linux 侧机械臂命令行入口，支持状态查询、使能、关节/位姿移动等命令。 |
| `kinematics.py` | Piper 正运动学辅助函数，基于 DH/变换矩阵计算末端位姿并处理欧拉角。 |
| `kinematics_test.py` | 正运动学相关单元测试。 |
| `motion.py` | 高层 robot-frame 运动辅助封装，提供 `MotionPlanner` 之类的规划入口。 |
| `safety.py` | 机械臂命令安全检查，包括工作空间、位姿范围等基础校验；定义 `SafetyError` 和 `SafetyChecker`。 |
| `state_monitor.py` | Piper 反馈消息读取和状态整理辅助类。 |

### 6.2 `robot_control/linux/config/`

Linux 控制侧配置。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `robot_config.py` | 定义 `ROBOT_CONFIG`、位姿限制、DH 参数和 CAN 名称等机器人配置。 |

### 6.3 `robot_control/linux/gripper/`

Piper 官方夹爪控制、调试和链路诊断。

| 文件 | 作用 |
| --- | --- |
| `README.md` | 夹爪手动控制说明，包括位置/力矩范围、单次控制、交互控制和常用参数。 |
| `__init__.py` | Python 包标识文件。 |
| `piper_gripper.py` | 夹爪便利封装类，面向上层提供更直接的夹爪控制接口。 |
| `gripper_config.yaml` | 夹爪标定/配置文件，记录命令百分比、命令宽度、反馈位置和实际宽度等校准点。 |
| `gripper_debug.py` | 夹爪手动调试工具，支持通信检查、状态读取、夹爪命令、清错、校准记录和 JSON/YAML 输出。 |
| `gripper_debug_test.py` | `gripper_debug.py` 的单元测试。 |
| `control_gripper.py` | 单独控制夹爪位置和力矩的 CLI，不修改零点，支持单次命令和交互模式。 |
| `control_gripper_test.py` | `control_gripper.py` 的单元测试。 |
| `diagnose_j6_gripper.py` | J6 驱动板到夹爪链路诊断工具，检查 SDK 反馈、J6 反馈、夹爪命令和 CAN 帧可见性。 |
| `diagnose_j6_gripper_test.py` | `diagnose_j6_gripper.py` 的单元测试。 |
| `gripper_test.py` | 夹爪封装层单元测试。 |

### 6.4 `robot_control/linux/server/`

通用 Linux TCP server。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `tcp_server.py` | 接收 Windows 相机点命令的 TCP server；可用临时映射生成机器人位姿，默认 dry-run，实机需显式 `--execute`。 |

### 6.5 `robot_control/linux/tcp_calibration/`

夹爪 TCP 固定点多姿态标定工具。这里标定的是 `J6 -> gripper_tcp` 平移偏移，不是相机外参。

| 文件 | 作用 |
| --- | --- |
| `README.md` | 夹爪 TCP 标定流程说明，包括采集、求解、独立验证和输出文件含义。 |
| `__init__.py` | Python 包标识文件。 |
| `calibration.py` | 固定点 pivot calibration 数学和 YAML 文件读写辅助；求解 TCP 偏移、评估误差、采集反馈位姿。 |
| `calibration_test.py` | TCP 标定数学和文件处理的单元测试。 |
| `collect_tcp_data.py` | 交互式采集 TCP 标定样本，只读取 Piper J6 反馈，不主动控制机械臂运动。 |
| `solve_tcp.py` | 从 `tcp_samples.yaml` 求解 `J6 -> gripper_tcp` 平移偏移并写入 `tcp.yaml`。 |
| `validate_tcp.py` | 使用独立样本验证已求解 TCP 标定，报告 RMS 和最大误差。 |
| `tcp.yaml` | 当前 TCP 标定结果文件；当前状态为 `uncalibrated`，字段预留 `x_mm/y_mm/z_mm`。 |

### 6.6 `robot_control/linux/hand_eye/`

Linux 侧手眼标定和样本管理。默认 `eye_to_hand` 模式输出 `camera -> base`，`eye_in_hand` 模式输出 `gripper -> camera`。

| 文件 | 作用 |
| --- | --- |
| `README.md` | Piper 手眼标定流程说明，包括 Linux 采样 server、Windows 棋盘采集、自动采集、求解和验证建议。 |
| `__init__.py` | Python 包标识文件。 |
| `calibration.py` | 手眼标定数学和 JSON 样本/结果文件辅助；封装 OpenCV `calibrateHandEye` 相关求解、残差评估和格式转换。 |
| `calibration_test.py` | 手眼标定数学单元测试。 |
| `collect_server.py` | Linux 侧手眼采样 server；接收 Windows 棋盘格位姿，同时读取当前 Piper 末端位姿并保存同步样本。 |
| `solve_hand_eye.py` | 从采集样本求解手眼标定结果，输出 `hand_eye_result.json` 或自动结果文件。 |
| `auto_collect_hand_eye.py` | 自动小步随机移动机械臂并请求 Windows 视觉服务检测棋盘格的自动采样脚本。 |
| `auto_collect_hand_eye_test.py` | 自动采样运动安全和关节跳变检查相关单元测试。 |
| `move_board_pose.py` | 移动夹爪上的标定板到指定位置/姿态的辅助脚本，便于人工采集手眼样本。 |
| `read_pose.py` | 打印当前 Piper 末端位姿，用于标定和现场记录。 |
| `hand_eye_samples.json` | 手眼人工采样样本文件，通常用于 `eye_to_hand` 求解。 |
| `hand_eye_result.json` | 当前默认手眼结果文件，`eye_to_hand` 下包含 `camera -> base` 变换，抓取 server 常用。 |
| `eye_to_hand_samples_auto.json` | 自动采样得到的 eye-to-hand 样本。 |
| `eye_to_hand_result_auto.json` | 自动样本求解得到的 eye-to-hand 结果，包含 `camera -> base` 变换。 |
| `eye_in_hand_samples_auto.json` | 自动采样得到的 eye-in-hand 样本。 |
| `eye_in_hand_result_auto.json` | 自动样本求解得到的 eye-in-hand 结果，包含 `gripper -> camera` 或等价变换信息。 |

### 6.7 `robot_control/linux/catch/`

Linux 抓取执行层，包含点击抓取、YOLO 传送带抓取、eye-in-hand 抓取、固定轨迹 demo、调参文档和测试。

| 文件 | 作用 |
| --- | --- |
| `README.md` | 点击抓取 server 使用说明，包含手眼结果、TCP 偏移、抓取流程、夹爪零点、固定脚本测试和常用调参。 |
| `__init__.py` | Python 包标识文件；包文档说明这里是 Linux 侧 click-to-catch server。 |
| `catch_server.py` | Linux 点击抓取 TCP server；接收 camera XYZ，使用手眼或临时映射生成抓取/放置计划并可执行机械臂和夹爪动作。 |
| `catch_server_test.py` | `catch_server.py` 的单元测试，覆盖 TCP 偏移、抓取计划和 fake SDK 行为。 |
| `catch_server_1.py` | 当前 YOLO 传送带抓取核心 server；接收 Windows YOLO/OpenCV camera point，加载 eye-to-hand 结果，生成 out/in 动作序列，支持动态时序、固定/动态 pitch、safe gripper、planar IK 等实机场景逻辑。 |
| `catch_server_1_test.py` | `catch_server_1.py` 的单元测试，覆盖目标校验、动态时序、夹爪保护、平面 IK、位姿序列等逻辑。 |
| `catch_server_2.py` | eye-in-hand 静态抓取 server；复用 `catch_server_1` 动作序列，但每条消息根据当前末端位姿计算 camera-to-base。 |
| `catch_server_coordinate_and_args_guide.md` | 抓取 server 坐标系、参数和现场调参说明。 |
| `catch_server_1_dynamic_pitch_guide.md` | `catch_server_1.py` 动态 pitch / 姿态调整相关说明。 |
| `hand_eye_mapping.py` | eye-to-hand `camera -> base` 映射辅助，负责加载手眼结果、质量摘要和 camera point 到 base 坐标转换。 |
| `eye_in_hand_mapping.py` | eye-in-hand 变换辅助，基于当前 gripper pose 计算 camera-to-base，并支持目标偏移。 |
| `eye_in_hand_mapping_test.py` | eye-in-hand 映射和 `catch_server_2` 参数桥接逻辑测试。 |
| `tcp_offset.py` | 官方 Piper 夹爪 TCP 偏移工具，支持 J6 pose 和 TCP pose 之间互算，并生成 `CatchPlan`。 |
| `debug_hand_eye_point.py` | 单点调试脚本，把指定 camera XYZ 通过手眼映射和抓取规划打印成 target/grasp/place/J6 位姿，便于检查坐标和边界。 |
| `set_gripper_zero.py` | 设置夹爪零点脚本，通常在物理完全张开位置执行，使后续 `0 mm` 表示完全张开。 |
| `set_gripper_zero_test.py` | `set_gripper_zero.py` 的单元测试。 |
| `scripted_catch_demo.py` | 不依赖相机输入的固定 top-down 抓取/放置 demo；先打印并校验 J6 waypoint，可 dry-run 或执行。 |
| `scripted_catch_demo_test.py` | `scripted_catch_demo.py` 的单元测试。 |
| `run_pose_continuous_catch_sequence.py` | 固定位姿连续抓取序列脚本，直接使用 robot-frame J6 pose，适合固定坐标搬运和连续动作验证。 |
| `run_pose_continuous_catch_sequence_test.py` | 连续抓取序列脚本的单元测试。 |
| `run_multiple_catch_sequence.py` | 按预设命令列表依次运行多次抓取命令的脚本，适合重复流程验证。 |

## 7. `robot_control/windows/`

Windows 视觉端。该层通常需要 RealSense D435i、OpenCV 窗口环境和 YOLO 模型。

| 文件/目录 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `vision/` | 相机、深度、检测器、YOLO、传送带深度平面和可视化基础模块。 |
| `client/` | 面向操作/联调的 Windows 客户端脚本，负责采集、检测、点击、发送 TCP。 |
| `hand_eye/` | Windows 侧手眼标定棋盘格检测、采样发送和点击验证。 |
| `task/` | 视觉侧任务生成预留包；当前只有包标识文件。 |

### 7.1 `robot_control/windows/vision/`

Windows 视觉基础模块。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `camera.py` | 相机模块占位文件；当前没有业务实现。 |
| `visualization.py` | 可视化模块占位文件；当前没有业务实现。 |
| `depth.py` | RealSense 深度工具，创建 RGB/depth 对齐流、读取深度、像素反投影为 camera-frame XYZ，并定义 `CameraPoint`。 |
| `depth_test.py` | 深度读取和反投影单元测试。 |
| `detector.py` | OpenCV 目标检测基础模块，定义 `Detection`、背景减除检测器、颜色阈值检测器和 mask 清理函数。 |
| `detector_test.py` | OpenCV 检测器单元测试。 |
| `yolo_detector.py` | Ultralytics YOLO 封装，负责模型路径、配置目录、模型加载和 RGB 目标检测。 |
| `yolo_detector_test.py` | YOLO 封装单元测试，使用 fake result 验证解析逻辑。 |
| `download_yolo_model.py` | 下载 YOLO26n 预训练模型到 `models/` 的脚本。 |
| `d435i_intrinsics.py` | 加载 D435i rectified 内参 JSON，并输出 `PinholeIntrinsics`。 |
| `d435i_intrinsics_test.py` | D435i 内参加载单元测试。 |
| `d435i_para.json` | D435i 相机内参/标定参数文件，供视觉端反投影使用。 |
| `belt_depth_plane.py` | 图像空间传送带深度平面拟合、保存、加载和补偿工具。 |
| `belt_depth_plane_test.py` | 传送带深度平面拟合与补偿单元测试。 |

### 7.2 `robot_control/windows/client/`

Windows 侧人工调试和视觉抓取客户端入口。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件。 |
| `realsense_rgb_preview.py` | RealSense RGB 预览脚本，用于检查相机连接和画面。 |
| `click_to_camera_xyz.py` | 手动点击 RGB 图，读取对齐深度并打印 camera XYZ，不发送给 Linux。 |
| `click_send_tcp.py` | 手动点击 RGB 图并将 camera XYZ 通过 TCP 发送到 Linux 抓取/运动 server。 |
| `yolo_detection_to_camera_xyz.py` | YOLO 检测后把 bbox 中心结合深度转为 camera XYZ，只做视觉定位调试。 |
| `yolo_grasp_send_tcp.py` | YOLO bbox 抓取客户端；检测目标、按深度排序、反投影、支持静态人工选择和动态传送带自动发送。 |
| `yolo_grasp_send_tcp_test.py` | YOLO 抓取客户端的单元测试。 |
| `yolo_seg_grasp_send_tcp.py` | YOLO segmentation 抓取客户端；基于 mask 质心/顶部点和 mask 深度统计生成抓取点，支持动态发送。 |
| `yolo_seg_grasp_send_tcp_test.py` | YOLO segmentation 抓取客户端单元测试。 |
| `opencv_grasp_send_tcp.py` | OpenCV 颜色/轮廓抓取客户端；面向传送带场景从 mask centroid、深度和目标短轴角度生成抓取命令。 |
| `opencv_grasp_send_tcp_test.py` | OpenCV 抓取客户端单元测试。 |
| `calibrate_belt_depth_plane.py` | 空传送带深度平面标定脚本，可自动/手动采样 ROI 并输出 `belt_depth_plane.json`。 |

### 7.3 `robot_control/windows/hand_eye/`

Windows 侧手眼标定采集和验证。

| 文件/目录 | 作用 |
| --- | --- |
| `README.md` | Windows 侧手眼标定采集说明。 |
| `__init__.py` | Python 包标识文件；包文档说明这里是 Windows 侧手眼采集工具。 |
| `chessboard.py` | 棋盘格检测和位姿估计基础模块，定义相机内参、棋盘配置、检测结果和绘制函数。 |
| `chessboard_test.py` | 棋盘格检测/内参处理单元测试。 |
| `capture_chessboard.py` | 人工采样脚本，检测棋盘格位姿并发送给 Linux `collect_server.py`。 |
| `capture_chessboard_auto_server.py` | 自动采样视觉服务，响应 Linux 自动采样请求，返回稳定的棋盘格检测位姿和质量指标。 |
| `validate_hand_eye_click.py` | 点击验证 eye-to-hand `camera -> base` 结果的脚本；点击已知 base 坐标点后打印映射误差。 |
| `validate_hand_eye_click_test.py` | 点击验证逻辑单元测试。 |
| `captures_auto/` | 自动手眼采样时保存的棋盘格图像，用于复查检测质量和样本来源。 |

#### `robot_control/windows/hand_eye/captures_auto/`

自动采样图像目录。文件名格式为 `auto-序号_日期_时间.png`，每张图对应一次自动采样保存的棋盘格画面。

| 文件 | 作用 |
| --- | --- |
| `auto-001_20260702_150454.png` | 自动手眼采样保存图像。 |
| `auto-001_20260702_150553.png` | 自动手眼采样保存图像。 |
| `auto-002_20260702_150510.png` | 自动手眼采样保存图像。 |
| `auto-002_20260702_150611.png` | 自动手眼采样保存图像。 |
| `auto-003_20260702_150532.png` | 自动手眼采样保存图像。 |
| `auto-003_20260702_150702.png` | 自动手眼采样保存图像。 |
| `auto-004_20260702_150817.png` | 自动手眼采样保存图像。 |
| `auto-005_20260702_150836.png` | 自动手眼采样保存图像。 |
| `auto-006_20260702_151115.png` | 自动手眼采样保存图像。 |
| `auto-007_20260702_151121.png` | 自动手眼采样保存图像。 |
| `auto-008_20260702_151230.png` | 自动手眼采样保存图像。 |
| `auto-009_20260702_151242.png` | 自动手眼采样保存图像。 |
| `auto-010_20260702_151327.png` | 自动手眼采样保存图像。 |
| `auto-011_20260702_151345.png` | 自动手眼采样保存图像。 |
| `auto-012_20260702_151504.png` | 自动手眼采样保存图像。 |
| `auto-013_20260702_151619.png` | 自动手眼采样保存图像。 |
| `auto-014_20260702_151657.png` | 自动手眼采样保存图像。 |
| `auto-015_20260702_151741.png` | 自动手眼采样保存图像。 |
| `auto-016_20260702_151759.png` | 自动手眼采样保存图像。 |
| `auto-017_20260702_151818.png` | 自动手眼采样保存图像。 |
| `auto-018_20260702_152310.png` | 自动手眼采样保存图像。 |
| `auto-019_20260702_152315.png` | 自动手眼采样保存图像。 |
| `auto-020_20260702_152340.png` | 自动手眼采样保存图像。 |
| `auto-021_20260702_152425.png` | 自动手眼采样保存图像。 |
| `auto-022_20260702_152443.png` | 自动手眼采样保存图像。 |
| `auto-023_20260702_152544.png` | 自动手眼采样保存图像。 |
| `auto-024_20260702_152549.png` | 自动手眼采样保存图像。 |
| `auto-025_20260702_152621.png` | 自动手眼采样保存图像。 |

### 7.4 `robot_control/windows/task/`

视觉端任务生成预留目录。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python 包标识文件；当前没有具体任务实现。 |

## 8. 生成目录和工具目录

这些目录/文件不是主要业务源码，但会出现在当前工作区。

| 文件/目录 | 作用 |
| --- | --- |
| `.git/` | Git 仓库元数据目录，不应手动编辑，文档不展开其内部文件。 |
| `.agents/` | 当前 Codex/代理运行环境相关目录，不属于业务源码。 |
| `.codex/` | 当前 Codex/代理运行环境相关目录，不属于业务源码。 |
| `.pytest_cache/` | pytest 运行缓存目录。 |
| `.pytest_cache/.gitignore` | pytest 缓存目录的忽略规则。 |
| `.pytest_cache/CACHEDIR.TAG` | 标记该目录为缓存目录的标准文件。 |
| `.pytest_cache/README.md` | pytest 自动生成的缓存说明。 |
| `.pytest_cache/v/cache/lastfailed` | pytest 上次失败测试缓存。 |
| `.pytest_cache/v/cache/nodeids` | pytest 收集到的测试节点缓存。 |
| `__pycache__/` | Python 字节码缓存目录，可删除后由解释器重新生成。 |
| `robot_control/**/__pycache__/` | 各 Python 包下的字节码缓存目录，可删除后由解释器重新生成。 |
| `*.pyc` | Python 编译缓存文件，不需要人工维护或作为源码阅读。 |

## 9. 主要运行入口速查

| 场景 | 入口 |
| --- | --- |
| 机械臂 CLI | `python3 -m robot_control.linux.arm.arm_cli ...` |
| 夹爪单独控制 | `python3 -m robot_control.linux.gripper.control_gripper ...` |
| 夹爪调试/校准 | `python3 -m robot_control.linux.gripper.gripper_debug ...` |
| J6/夹爪链路诊断 | `python3 -m robot_control.linux.gripper.diagnose_j6_gripper ...` |
| Linux 通用 TCP server | `python3 -m robot_control.linux.server.tcp_server ...` |
| Linux 点击抓取 server | `python3 -m robot_control.linux.catch.catch_server ...` |
| Linux YOLO 传送带抓取 server | `python3 -m robot_control.linux.catch.catch_server_1 ...` |
| Linux eye-in-hand 抓取 server | `python3 -m robot_control.linux.catch.catch_server_2 ...` |
| 固定抓取 demo | `python3 -m robot_control.linux.catch.scripted_catch_demo ...` |
| 固定位姿连续抓取 | `python3 -m robot_control.linux.catch.run_pose_continuous_catch_sequence ...` |
| Windows RGB 预览 | `python -m robot_control.windows.client.realsense_rgb_preview` |
| Windows 点击相机坐标 | `python -m robot_control.windows.client.click_to_camera_xyz` |
| Windows 点击发送 TCP | `python -m robot_control.windows.client.click_send_tcp ...` |
| Windows YOLO 目标定位 | `python -m robot_control.windows.client.yolo_detection_to_camera_xyz` |
| Windows YOLO 抓取发送 | `python -m robot_control.windows.client.yolo_grasp_send_tcp ...` |
| Windows YOLO 分割抓取发送 | `python -m robot_control.windows.client.yolo_seg_grasp_send_tcp ...` |
| Windows OpenCV 抓取发送 | `python -m robot_control.windows.client.opencv_grasp_send_tcp ...` |
| 手眼采样 Linux server | `python3 -m robot_control.linux.hand_eye.collect_server ...` |
| 手眼采样 Windows 客户端 | `python -m robot_control.windows.hand_eye.capture_chessboard ...` |
| 自动手眼采样 Linux 端 | `python3 -m robot_control.linux.hand_eye.auto_collect_hand_eye ...` |
| 自动手眼采样 Windows 视觉服务 | `python -m robot_control.windows.hand_eye.capture_chessboard_auto_server ...` |
| 手眼求解 | `python3 -m robot_control.linux.hand_eye.solve_hand_eye ...` |
| 手眼点击验证 | `python -m robot_control.windows.hand_eye.validate_hand_eye_click ...` |
| TCP 标定采集 | `python3 -m robot_control.linux.tcp_calibration.collect_tcp_data ...` |
| TCP 标定求解 | `python3 -m robot_control.linux.tcp_calibration.solve_tcp ...` |
| TCP 标定验证 | `python3 -m robot_control.linux.tcp_calibration.validate_tcp ...` |

## 10. 测试文件命名约定

工程中的测试文件基本与被测模块放在同一目录，命名为 `*_test.py`。推荐从项目根目录运行：

```bash
python3 -m unittest discover -s robot_control -p '*test.py'
```

涉及 RealSense、OpenCV 窗口、Piper SDK/CAN 和真实机械臂的脚本一般是手动/硬件联调入口，不属于无硬件自动测试范围。
