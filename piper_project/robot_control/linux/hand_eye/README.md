# Piper 手眼标定

这里实现两端协作的手眼标定采样流程：

```text
机械臂移动
  -> Windows 相机拍照
  -> 计算棋盘格 target -> camera 位姿
  -> Linux 读取 Piper 末端 base -> gripper 位姿
  -> 保存一组样本
  -> 重复 20-30 次
  -> OpenCV calibrateHandEye()
```

默认模式是 `eye_to_hand`：相机固定，夹爪夹着棋盘格移动。求解结果是
`camera -> base`。如果相机装在末端、棋盘格固定，采集和求解时改成
`--mode eye_in_hand`，结果是 `gripper -> camera`。

## 棋盘格和相机参数

默认参数：

- 棋盘格内角点：`11 x 8`
- 格子边长：`15 mm`
- 相机内参：RealSense rectified `640 x 480`
  - `fx=391.185`
  - `fy=391.185`
  - `cx=319.123`
  - `cy=244.756`
  - 畸变默认按 0 处理

注意：OpenCV 的棋盘格尺寸是“内角点数量”，不是黑白方格数量。如果你的
“8 x 11 棋盘格”指的是 8 x 11 个方格，应在 Windows 采集端使用
`--corners-rows 7 --corners-cols 10`。

## 1. Linux 启动采样服务器

在 Linux / VM 的项目根目录运行：

```bash
python3 -m robot_control.linux.hand_eye.collect_server \
  --host 0.0.0.0 \
  --port 5010 \
  --can can0 \
  --mode eye_to_hand \
  --overwrite
```

服务器只读取机械臂反馈，不主动移动机械臂。每收到 Windows 的一组棋盘格位姿，
它会读取当前末端位姿并追加到：

```text
robot_control/linux/hand_eye/hand_eye_samples.json
```

## 2. 移动夹爪上的标定板

只改姿态、保持当前 XYZ，先 dry-run：

```bash
python3 -m robot_control.linux.hand_eye.move_board_pose \
  --roll-deg 10 \
  --pitch-deg 75 \
  --yaw-deg -20
```

确认目标安全后执行：

```bash
python3 -m robot_control.linux.hand_eye.move_board_pose \
  --roll-deg 10 \
  --pitch-deg 75 \
  --yaw-deg -20 \
  --speed 10 \
  --execute
```

也可以同时指定 XYZ：

```bash
python3 -m robot_control.linux.hand_eye.move_board_pose \
  --x-mm 230 \
  --y-mm -40 \
  --z-mm 280 \
  --roll-deg 0 \
  --pitch-deg 80 \
  --yaw-deg 30 \
  --speed 10 \
  --execute
```

## 3. Windows 采集棋盘格

在 Windows 项目根目录运行，将 `<linux-ip>` 换成 Linux/VM 的 IP：

```bash
python -m robot_control.windows.hand_eye.capture_chessboard ^
  --host <linux-ip> ^
  --port 5010 ^
  --width 640 ^
  --height 480 ^
  --intrinsic-profile 640x480 ^
  --corners-rows 8 ^
  --corners-cols 11 ^
  --checker-size-mm 15
```

窗口中看到棋盘格角点和坐标轴后，按 `c` 采集并发送一组样本。按 `Esc` 退出。

如果用 1280 x 720：

```bash
python -m robot_control.windows.hand_eye.capture_chessboard ^
  --host <linux-ip> ^
  --width 1280 ^
  --height 720 ^
  --intrinsic-profile 1280x720 ^
  --corners-rows 8 ^
  --corners-cols 11 ^
  --checker-size-mm 15
```

## 4. 实际采集建议

采集 20-30 组。每组操作：

1. Linux 侧用 `move_board_pose` 或手动控制机械臂，让棋盘格出现在相机视野内。
2. 等机械臂完全停稳。
3. Windows 画面确认棋盘格完整、角点稳定、坐标轴方向正常。
4. Windows 按 `c`。
5. Linux 终端确认出现 `Saved sample N`。
6. 改变姿态和位置，重复。

样本要有明显的姿态变化，尤其是 roll/pitch/yaw 都要覆盖，不要只平移。建议：

- 棋盘格距离相机有近有远。
- 棋盘格在画面中心、左上、右上、左下、右下都出现过。
- 每次角度变化 10-25 度，避免角点太斜导致识别不稳。
- 不要让棋盘格部分出画、反光、模糊或被夹爪遮挡。

## 5. 自动采集

自动采集脚本会主动小范围移动机械臂、查询 Windows 视觉服务，并保存合格样本。
默认使用关节空间小步随机移动，避免 `EndPoseCtrl` 的 IK 分支切换导致腕部突然
翻转。

```bash
python3 -m robot_control.linux.hand_eye.auto_collect_hand_eye \
  --vision-host <windows-ip> \
  --vision-port 5011 \
  --can can0 \
  --mode eye_to_hand \
  --output robot_control/linux/hand_eye/eye_to_hand_samples_auto.json \
  --samples 25 \
  --speed 5 \
  --post-move-stabilize-s 4.0 \
  --overwrite \
  --execute
```

安全相关默认值：

- `--motion-space joint`：默认值，使用 `JointCtrl` 小步采样。
- `--step-base-joint-deg 3`、`--step-wrist-joint-deg 4`：每次随机候选的最大步长。
- `--range-base-joint-deg 15`、`--range-wrist-joint-deg 18`：相对 seed 的最大采样范围。
- `--max-joint-delta-deg 12`、`--max-wrist-joint-delta-deg 8`：单段运动的反馈跳变上限。

脚本会在每段运动期间读取关节反馈；如果 J4/J5/J6 等关节跳变超过阈值，会停止
自动采集并要求人工确认安全姿态。不要在现场使用 `--motion-space pose`，除非只是
复现实验；该模式仍有跳变监控，但无法从源头避免 IK 换解。

## 6. 求解

Linux 侧运行：

```bash
python3 -m robot_control.linux.hand_eye.solve_hand_eye \
  --samples robot_control/linux/hand_eye/hand_eye_samples.json \
  --output robot_control/linux/hand_eye/hand_eye_result.json \
  --mode eye_to_hand \
  --method tsai
```

输出文件：

```text
robot_control/linux/hand_eye/hand_eye_result.json
```

`eye_to_hand` 下结果 `transform` 是 `camera -> base`。后续把相机坐标系下的点
变换到机械臂基坐标系时使用这个矩阵。

## 7. 验证矩阵

矩阵准确性的直接验证方式是：在 Windows 相机画面里点击一个 base 坐标已知的
物理点，用 `camera -> base` 反推它的机械臂基坐标，然后和真实坐标比较。

Windows 项目根目录运行：

```bat
python -m robot_control.windows.hand_eye.validate_hand_eye_click ^
  --hand-eye-result robot_control\linux\hand_eye\hand_eye_result.json ^
  --expected-base-mm X Y Z
```

例如点击机械臂真实 base 原点时才使用：

```bat
--expected-base-mm 0 0 0
```

如果机械臂底座原点在画面中不可直接点击，建议在工作台上选一个小标记，先用
机械臂 TCP 对准该点并读取它在 base 系下的坐标，再把这个坐标填入
`--expected-base-mm`。脚本会打印点击点的相机坐标、反推 base 坐标，以及误差
范数。多点几次看误差是否稳定；若误差方向和大小明显漂移，优先排查深度噪声、
内参分辨率、相机是否移动、标定板固定是否松动。

如果残差很大，优先检查：

- 棋盘格尺寸是否用的是内角点数量。
- 相机分辨率和内参是否一致。
- 每次按 `c` 时机械臂是否已经停稳。
- 标定板是否真的刚性固定在夹爪上。
- 采样姿态是否只有平移、缺少旋转变化。
