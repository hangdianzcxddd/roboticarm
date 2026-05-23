# Piper Project

## 当前测试范围

目前只实现并测试以下链路：

```text
机械臂坐标系 -> Piper SDK -> 机械臂运动
```

视觉识别、深度计算、手眼标定、分拣任务逻辑暂时留空。

## 环境准备

进入工程目录：

```bash
cd /home/fishros/projects/roboticarm/piper_project
```

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

确认 Piper CAN 口已经配置好，例如使用 `can0`。默认配置在：

```text
config/robot_config.py
```

如果你的 CAN 口不是 `can0`，修改 `RobotConfig.can_name`。

## 运行单元测试

这些测试不需要连接真实机械臂，使用的是 mock SDK：

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

## 读取机械臂状态

连接机械臂但不使能，只读取状态、末端位姿和关节状态：

```bash
python3 main.py status
```

## 使能机械臂

```bash
python3 main.py enable
```

## 失能机械臂

```bash
python3 main.py disable
```

## 测试末端位姿运动

单位说明：

```text
x y z: mm
rx ry rz: degree
speed: 0-100
mode: P 或 L
```

示例：

```bash
python3 main.py pose 150 0 215 0 85 0 --speed 20 --mode P
```

直线运动示例：

```bash
python3 main.py pose 150 50 215 0 85 0 --speed 20 --mode L
```

## 测试关节运动

单位说明：

```text
j1-j6: rad
speed: 0-100
```

示例：

```bash
python3 main.py joints 0 0 -0.5 0 0.5 0 --speed 20
```

## 安全注意

运行真实机械臂前，请先确认：

```text
1. 机械臂工作空间内没有障碍物。
2. 急停按钮可用。
3. CAN 口配置正确。
4. config/robot_config.py 中的坐标范围、关节范围、默认速度符合当前安装环境。
5. 第一次测试建议使用较低速度，例如 --speed 10 或 --speed 20。
```

## 视觉模块

待补充。

## 标定模块

待补充。

## 分拣任务

待补充。
