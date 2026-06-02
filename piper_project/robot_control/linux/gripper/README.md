# Piper 夹爪手动控制

`control_gripper.py` 用于单独调节夹爪位置和扭矩。它只发送普通夹爪控制命令，不会修改夹爪零点。

如果已经把完全张开位置设置为零点：

- `0 mm` 表示完全张开。
- 负数表示向闭合方向运动，例如 `-30 mm`。
- 当前软件限位为 `[-100, 100] mm`。
- 扭矩范围为 `[0, 5] N.m`。

从项目根目录运行单次控制：

```bash
cd /home/zcxddd/projects/roboticarm/piper_project

python3 -m robot_control.linux.gripper.control_gripper \
  --can can0 \
  --command-mm -30 \
  --effort-nm 1.0
```

保持连接并连续输入不同目标：

```bash
python3 -m robot_control.linux.gripper.control_gripper \
  --can can0 \
  --interactive \
  --effort-nm 1.0
```

交互模式输入示例：

```text
gripper> 0 0.4
gripper> -20 0.8
gripper> -50 1.2
gripper> status
gripper> quit
```

常用参数：

- `--clear-error`：首次发送普通控制命令时同时清除夹爪错误，不改变零点。
- `--duration-s 1.0 --period-s 0.02`：在指定时长内重复发送目标命令。
- `--no-feedback`：不打印夹爪反馈。
