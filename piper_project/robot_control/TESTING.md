# Test And Manual Script Reference

## Automated Tests

Run tests that do not require hardware:

```bash
python3 -m unittest discover -s robot_control -p '*test.py'
```

Important test modules:

- `robot_control/linux/arm/piper_arm_test.py`
- `robot_control/linux/gripper/gripper_test.py`
- `robot_control/linux/gripper/gripper_debug_test.py`
- `robot_control/shared/protocol_test.py`
- `robot_control/windows/vision/detector_test.py`
- `robot_control/windows/vision/yolo_detector_test.py`

## Model Download

Download YOLO26n:

```bash
python -m robot_control.windows.vision.download_yolo_model
```

With explicit URL or output:

```bash
python -m robot_control.windows.vision.download_yolo_model \
  --url https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
  --output models/yolo26n.pt
```

## RealSense Manual Scripts

These require a RealSense camera and open OpenCV windows. Press Esc to exit.

RGB preview:

```bash
python -m robot_control.windows.client.realsense_rgb_preview
```

Click RGB point and print camera XYZ:

```bash
python -m robot_control.windows.client.click_to_camera_xyz
```

OpenCV detection to camera XYZ:

```bash
python -m robot_control.windows.client.opencv_detection_to_camera_xyz
```

YOLO detection to camera XYZ:

```bash
python -m robot_control.windows.client.yolo_detection_to_camera_xyz
```

## Windows To Linux TCP Test

Linux VM:

```bash
python3 -m robot_control.linux.server.tcp_server --host 0.0.0.0 --port 5005
```

Windows host:

```bash
python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005
```

Local protocol loopback without RealSense:

```bash
python -c "from robot_control.shared.protocol import CameraPointCommand, send_camera_point; print(send_camera_point('127.0.0.1', 5005, CameraPointCommand(x_m=0.1, y_m=-0.05, z_m=0.4, u=320, v=240, depth_m=0.4)))"
```
