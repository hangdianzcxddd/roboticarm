# Test Scripts

本文档记录 `test` 目录下所有非空测试脚本的用途和运行方式。

默认从项目目录运行：

```powershell
cd D:\git\projects\roboticarm\piper_project
```

## 自动单元测试

这些测试不需要连接真实机械臂或 RealSense。

### test_move.py

用途：测试 Piper 机械臂控制层的运动指令、安全边界、单位转换和 SDK 调用参数。

运行：

```powershell
python -m unittest test.test_move
```

覆盖内容：

- 末端位姿运动 `move_to_pose`
- 关节运动 `move_joints`
- 角度单位转换：rad -> milli-degree
- 位姿和关节安全范围校验
- 连接、使能、断开流程
- 安全失能前停车
- 碰撞保护等级配置

### test_gripper.py

用途：测试夹爪控制逻辑和 SDK 参数转换。

运行：

```powershell
python -m unittest test.test_gripper
```

覆盖内容：

- 打开/关闭夹爪
- 夹爪宽度单位转换：mm -> micrometer
- 夹爪力矩范围校验
- 清除夹爪错误码

### test_detection.py

用途：测试 OpenCV 目标检测算法，不需要相机。

运行：

```powershell
python -m unittest test.test_detection
```

覆盖内容：

- 从二值 mask 提取轮廓
- 计算目标中心点 `(u, v)`
- 过滤小面积噪声
- MOG2 背景检测
- HSV 颜色阈值检测

### test_yolo_detection.py

用途：测试 YOLO 检测结果解析逻辑，不需要相机，也不跑真实模型推理。

运行：

```powershell
python -m unittest test.test_yolo_detection
```

覆盖内容：

- YOLO bbox 转项目通用 `Detection`
- bbox center 计算 `(u, v)`
- 类别名称、类别 id、置信度解析

### test_tcp_protocol.py

用途：测试 Windows 视觉端和 Linux 控制端之间的 TCP JSON-lines 协议，以及未标定阶段的模拟坐标映射。

运行：

```powershell
python -m unittest test.test_tcp_protocol
```

覆盖内容：

- 相机坐标消息编码/解码
- `camera_point` 消息字段解析
- 相机坐标到模拟机械臂位姿的幅度限制

### 推荐自动测试命令

```powershell
python -m unittest test.test_move test.test_gripper test.test_detection test.test_yolo_detection test.test_tcp_protocol
```

不要直接用下面这种方式跑全部文件：

```powershell
python -m unittest discover -s test -p "test_*.py"
```

原因：`test_camera.py` 是顶层相机脚本，被 `unittest` 导入时会直接启动 RealSense 窗口。

## 模型文件

当前 YOLO26n 预训练模型已经保存到：

```text
models\yolo26n.pt
```

当前项目还会把 Ultralytics 运行配置保存在：

```text
models\Ultralytics
```

这样可以避免 Ultralytics 写入用户目录，例如 `AppData\Roaming\Ultralytics`。

### download_yolo_model.py

用途：下载 YOLO26n 预训练模型到 `models` 文件夹。

运行：

```powershell
python test\download_yolo_model.py
```

指定下载地址或保存路径：

```powershell
python test\download_yolo_model.py --url https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt --output models\yolo26n.pt
```

如果 `models\yolo26n.pt` 已经存在，脚本会直接跳过下载。

## RealSense 手动测试

这些脚本需要连接 RealSense，并会打开 OpenCV 窗口。按 `Esc` 退出窗口。

### test_camera.py

用途：最基础的 RealSense RGB 流测试，只验证彩色图像是否能正常显示。

运行：

```powershell
python test\test_camera.py
```

行为：

- 启动 `640x480`、`15fps` 的 RGB 流
- 显示 `RGB` 窗口
- 按 `Esc` 退出

### test_click_to_camera_xyz.py

用途：鼠标点击 RGB 图像，读取对齐后的深度，并输出相机坐标系下的 `X/Y/Z`。

运行：

```powershell
python test\test_click_to_camera_xyz.py
```

行为：

- 启动 RGB 和 Depth 流
- 将 Depth 对齐到 RGB
- 左键点击图像得到 `(u, v)`
- 读取点击点附近的 depth
- 反投影得到相机坐标 `X/Y/Z`
- 显示 RGB 窗口和深度有效区域窗口

常用参数：

```powershell
python test\test_click_to_camera_xyz.py --sample-radius 6
```

`--sample-radius` 用点击点周围有效深度的中位数，默认是 `3`。深度空洞多时可以调大，但过大会取到旁边物体的深度。

```powershell
python test\test_click_to_camera_xyz.py --hide-depth
```

隐藏深度调试窗口。

### test_opencv_detection_to_camera_xyz.py

用途：用 OpenCV 算法自动识别目标，取目标中心点 `(u, v)`，再读取 depth 并转换成相机坐标 `X/Y/Z`。

默认使用 MOG2 背景检测：

```powershell
python test\test_opencv_detection_to_camera_xyz.py
```

使用流程：

1. 启动后先保持目标不在画面中，让脚本学习背景。
2. 学习完成后把目标放入画面。
3. 程序会框出最大前景目标，并输出 `u/v/depth/X/Y/Z`。
4. 按 `r` 重新学习背景。
5. 按 `Esc` 退出。

常用参数：

```powershell
python test\test_opencv_detection_to_camera_xyz.py --min-area 1500
```

过滤小面积噪声，只保留面积较大的目标。

```powershell
python test\test_opencv_detection_to_camera_xyz.py --sample-radius 6
```

扩大深度采样邻域，减少中心点深度为 0 的情况。

```powershell
python test\test_opencv_detection_to_camera_xyz.py --warmup-frames 90
```

增加背景学习帧数，适合画面初始阶段不稳定的情况。

```powershell
python test\test_opencv_detection_to_camera_xyz.py --algorithm color
```

使用 HSV 颜色阈值检测，默认检测绿色范围。

```powershell
python test\test_opencv_detection_to_camera_xyz.py --algorithm color --lower-hsv 35 80 80 --upper-hsv 85 255 255
```

自定义 HSV 检测范围。

```powershell
python test\test_opencv_detection_to_camera_xyz.py --hide-mask --hide-depth
```

隐藏 mask 和深度调试窗口。

### test_yolo_detection_to_camera_xyz.py

用途：使用 YOLO26n 预训练模型识别目标，取 bbox center `(u, v)`，读取对齐后的 depth，并转换成相机坐标 `X/Y/Z`。

运行：

```powershell
python test\test_yolo_detection_to_camera_xyz.py
```

行为：

- 加载 `models\yolo26n.pt`
- 启动 RealSense RGB 和 Depth 流
- 将 Depth 对齐到 RGB
- YOLO 检测目标 bbox
- 取 bbox center 作为 `(u, v)`
- 读取中心点附近 depth
- 反投影得到相机坐标 `X/Y/Z`
- 显示 RGB 检测窗口和深度有效区域窗口

常用参数：

```powershell
python test\test_yolo_detection_to_camera_xyz.py --conf 0.4
```

提高置信度阈值，减少误检。

```powershell
python test\test_yolo_detection_to_camera_xyz.py --classes 39
```

只检测 COCO 类别 id 为 `39` 的目标，例如 `bottle`。

```powershell
python test\test_yolo_detection_to_camera_xyz.py --max-results 1
```

只输出置信度最高的一个目标。

```powershell
python test\test_yolo_detection_to_camera_xyz.py --sample-radius 8
```

扩大深度采样邻域，减少 bbox center 落在深度空洞时无法输出 XYZ 的情况。

```powershell
python test\test_yolo_detection_to_camera_xyz.py --device cpu
```

强制使用 CPU 推理。

```powershell
python test\test_yolo_detection_to_camera_xyz.py --hide-depth
```

隐藏深度调试窗口。

## Windows 到 Linux TCP 通信测试

目标架构：

```text
Windows RealSense 视觉
-> 点击图像得到相机坐标 XYZ
-> TCP 发送到 Linux 虚拟机
-> Linux 生成模拟机械臂位姿
-> dry-run 或调用 PiperController 执行运动
```

当前还没有空间点对应标定，因此 Linux 端使用 `calibration/simulated_mapping.py` 做保守模拟映射。这个映射不是实际相机到机械臂的标定结果，只用于通信链路和小幅度运动测试。

### tcp_motion_server.py

用途：Linux 端 TCP 服务。接收 Windows 端发来的相机坐标，生成一个保守的机械臂末端位姿。

Linux 虚拟机上运行：

```powershell
python test\tcp_motion_server.py --host 0.0.0.0 --port 5005
```

默认是 dry-run，不会移动机械臂，只会打印生成的位姿并返回 ack。

确认位姿安全后，才使用真实运动：

```powershell
python test\tcp_motion_server.py --host 0.0.0.0 --port 5005 --execute --speed 10
```

常用安全参数：

```powershell
python test\tcp_motion_server.py --max-delta-x-mm 20 --max-delta-y-mm 20 --max-delta-z-mm 15
```

限制由相机坐标引起的最大位移幅度。

```powershell
python test\tcp_motion_server.py --base-x-mm 150 --base-y-mm 0 --base-z-mm 220
```

设置模拟运动围绕的机械臂基准位姿。

```powershell
python test\tcp_motion_server.py --rx-deg 0 --ry-deg 85 --rz-deg 0
```

设置末端姿态角。

### test_click_send_tcp.py

用途：Windows 端 RealSense 点击测试。点击图像得到相机坐标后，通过 TCP 发送给 Linux 端 `tcp_motion_server.py`。

Windows 上运行：

```powershell
python test\test_click_send_tcp.py --host <linux-vm-ip> --port 5005
```

行为：

- 启动 RealSense RGB 和 Depth 流
- 将 Depth 对齐到 RGB
- 左键点击得到 `(u, v)` 和相机坐标 `X/Y/Z`
- 通过 TCP 发送到 Linux 虚拟机
- 打印 Linux 端返回的 ack 或错误信息

常用参数：

```powershell
python test\test_click_send_tcp.py --host <linux-vm-ip> --port 5005 --sample-radius 6
```

扩大深度采样邻域。

```powershell
python test\test_click_send_tcp.py --host <linux-vm-ip> --timeout-s 5
```

增加 TCP 响应等待时间。

### 本机回环通信测试

不连接 RealSense、不移动机械臂时，可以先在同一台机器上测试 TCP 协议：

终端 1：

```powershell
python test\tcp_motion_server.py --host 127.0.0.1 --port 5005
```

终端 2：

```powershell
python -c "from communication.protocol import CameraPointCommand, send_camera_point; print(send_camera_point('127.0.0.1', 5005, CameraPointCommand(x_m=0.1, y_m=-0.05, z_m=0.4, u=320, v=240, depth_m=0.4)))"
```

期望返回 `ack`，其中包含 `executed=False` 和生成的模拟机械臂位姿。

## 脚本选择建议

- 只确认相机 RGB 是否工作：运行 `test_camera.py`
- 验证点击点到相机坐标：运行 `test_click_to_camera_xyz.py`
- 验证 OpenCV 自动识别目标并输出 XYZ：运行 `test_opencv_detection_to_camera_xyz.py`
- 验证 YOLO26n 自动识别目标并输出 XYZ：运行 `test_yolo_detection_to_camera_xyz.py`
- 验证 Windows 到 Linux TCP 通信：Linux 运行 `tcp_motion_server.py`，Windows 运行 `test_click_send_tcp.py`
- 验证代码逻辑不依赖硬件：运行 `test_move.py`、`test_gripper.py`、`test_detection.py`、`test_yolo_detection.py`、`test_tcp_protocol.py`
