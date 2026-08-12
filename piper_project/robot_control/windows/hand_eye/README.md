# Windows 手眼标定采集

本目录负责相机端：

- 打开 RealSense 彩色流。
- 检测棋盘格。
- 用 OpenCV `solvePnP()` 计算 `target -> camera`。
- 按 `c` 发送样本到 Linux。

默认参数来自你提供的 RealSense rectified `640 x 480` 内参：

```text
fx=391.185 fy=391.185 cx=319.123 cy=244.756
```

运行：

```bat
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

按键：

- `c`：当前检测到棋盘格时采集并发送一组样本。
- `Esc`：退出。

OpenCV 的 `--corners-rows/--corners-cols` 是内角点数量。如果你的 8 x 11
描述的是方格数量，不是内角点数量，请改为：

```bat
--corners-rows 7 --corners-cols 10
```

## 验证 hand_eye_result.json

`eye_to_hand` 求解出的 `hand_eye_result.json` 是 `camera -> base`。验证时，
需要点击一个“在机械臂 base 坐标系里坐标已知”的物理点：

```bat
python -m robot_control.windows.hand_eye.validate_hand_eye_click ^
  --hand-eye-result robot_control\linux\hand_eye\hand_eye_result.json ^
  --expected-base-mm 0 0 0
```

只有当你点击的点确实是机械臂 base 原点时，才使用 `0 0 0`。更稳妥的做法是
在工作台上放一个坐标已知的小标记，或让 TCP 对准该点后读取它的 base 坐标，再
把这个坐标传给 `--expected-base-mm X Y Z`。左键点击后脚本会打印：

- 相机坐标 `camera XYZ`
- 矩阵反推的机械臂基坐标 `base XYZ`
- 和 `--expected-base-mm` 的误差

如果不传 `--expected-base-mm`，脚本只输出每次点击映射出的 base 坐标：

```bat
python -m robot_control.windows.hand_eye.validate_hand_eye_click
```

按 `Esc` 退出。可加 `--log-jsonl hand_eye_click_validation.jsonl` 保存每次点击。
