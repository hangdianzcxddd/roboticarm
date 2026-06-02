# Piper 夹爪 TCP 标定

这里实现的是夹爪 TCP 平移偏移标定，不是相机外参或手眼标定。它求解：

```text
J6 坐标系 -> 夹爪工作点 TCP
```

方法是固定点多姿态法（pivot calibration）。让同一个夹爪工作点在多个姿态下始终对准同一个固定参考点。脚本读取 Piper SDK 上报的 J6 位姿，求解 TCP 在 J6 坐标系中的 `x/y/z` 偏移。

## 能力边界

- 可以标定 TCP 平移偏移 `x/y/z`。
- 不会自动控制机械臂运动。采集时只读取反馈，移动机械臂需要使用示教器。
- 不标定 TCP 朝向。输出中 `rx/ry/rz = 0` 表示 TCP 坐标轴默认与 J6 坐标轴对齐。
- 不标定相机坐标系到机器人坐标系的变换。视觉抓取仍需单独实现相机外参或手眼标定。

## 准备

1. 确认急停可用，清空工作空间，使用低速示教。
2. 固定一个尖锐且不会移动的参考点，例如稳固安装的锥尖。
3. 明确要标定的夹爪工作点。固定点法标定的是实际对准参考点的那个点。
4. 如果目标是夹爪中心，使用能准确表示夹爪中心的治具。仅仅夹住一个长探针时，得到的是探针尖端 TCP，不会自动变成夹爪中心。
5. 在项目根目录执行命令：

```bash
cd /home/zcxddd/projects/roboticarm/piper_project
```

## 采集

连接 CAN 后运行：

```bash
python3 -m robot_control.linux.tcp_calibration.collect_tcp_data \
  --can can0 \
  --overwrite
```

脚本不会使能机械臂，也不会下发运动指令。使用示教器完成以下操作：

1. 将目标 TCP 对准固定参考点。
2. 回到终端按 Enter 记录当前位姿。
3. 保持 TCP 仍对准同一个参考点，改变末端姿态后再次记录。
4. 至少记录 6 组，建议记录 8 到 12 组。姿态应覆盖不同方向，不要只绕单一轴旋转。
5. 输入 `u` 可撤销上一组，输入 `q` 结束并写入 `tcp_samples.yaml`。

## 求解

```bash
python3 -m robot_control.linux.tcp_calibration.solve_tcp
```

默认要求拟合 RMS 误差不超过 `1.0 mm`，最大误差不超过 `2.0 mm`。通过后，结果写入 `tcp.yaml`，关键字段为：

```yaml
transform:
  from_frame: j6
  to_frame: gripper_tcp
  x_mm: ...
  y_mm: ...
  z_mm: ...
```

如果输出状态为 `poor_fit`，不要使用结果。重新确认参考点是否牢固、每次是否对准了同一个工作点，并增加姿态差异。

## 独立验证

不要复用求解时的样本。重新摆出至少 4 个姿态：

```bash
python3 -m robot_control.linux.tcp_calibration.validate_tcp \
  --collect \
  --can can0 \
  --overwrite
```

每个姿态仍需让相同 TCP 对准原固定参考点。输入 `q` 后脚本会报告验证 RMS 与最大误差。默认通过阈值同样是 `1.0 mm` 和 `2.0 mm`。

## 输出文件

- `tcp_samples.yaml`：求解样本，采集脚本生成。
- `tcp.yaml`：标定结果，求解脚本生成。
- `tcp_validation_samples.yaml`：独立验证样本，验证脚本使用 `--collect` 时生成。

这些结果当前不会自动注入运动控制或视觉抓取链路。接入抓取逻辑时，应显式加载 `tcp.yaml` 并应用 `J6 -> gripper_tcp` 偏移。
