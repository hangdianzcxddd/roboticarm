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
