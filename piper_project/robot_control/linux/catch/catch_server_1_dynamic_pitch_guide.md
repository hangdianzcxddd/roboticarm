# catch_server_1 距离自适应夹爪姿态说明

本文说明 `catch_server_1.py` 中“根据抓取物体距离调整夹爪姿态”的原理和实现做法。当前实现对应参数开关为 `--dynamic-pitch-by-x`，也可以使用语义更明确的别名 `--adaptive-joint-pitch-by-x`。

## 1. 功能目标

`catch_server_1.py` 的主流程仍然使用固定 J6 位姿序列：所有抓取、抬升和放置 waypoint 都保持同一组 `rx/ry/rz`。改进后的 dynamic pitch 不再“只要开启就按 X 连续补偿”，而是先判断目标是否处于固定夹爪姿态可抓区域：

```text
pitch_source_x_mm 在 keep 区间内
  -> mode=keep_fixed
  -> pitch_delta_deg=0
  -> 不标记 joint_pitch，不调用 JointCtrl
  -> 完全走原固定姿态动作逻辑

pitch_source_x_mm 在 keep 区间外
  -> mode=adjust
  -> 按超出 keep 边界的距离计算 pitch_delta
  -> 业务角度限幅
  -> J2/J3/J5 关节限位二次夹紧
  -> 通过 JointCtrl 叠加关节空间增量
```

这里的“夹爪姿态变化”仍然只由 J2/J3/J5 的关节空间增量实现，不修改任何 waypoint 的 `rx_deg`、`ry_deg`、`rz_deg`。

## 2. X 距离来源

在当前抓取场景里，物体主要沿机器人前后方向变化。映射到机器人基座坐标系后，`pick_x_mm` 越大，通常表示物体离机械臂基座越远；`pick_x_mm` 越小，通常表示物体更近。因此代码用目标 X 坐标近似驱动夹爪 pitch。

如果命令来自普通视觉点，X 来源是手眼标定后的 `base_xyz_m[0] * 1000`。如果命令来自动态抓取模式，则优先使用 Windows 侧传来的 `pick_x_mm`。

如果启用了 `--force-pick-x-mm`，最终抓取会走固定 X 线，但代码仍保留原始 X：

```text
pitch_source_x_mm = raw_pick_x_mm if raw_pick_x_mm exists else pick_x_mm
```

这样固定抓取线不会丢失真实远近信息。

## 3. 稳定区和角度计算

新增固定姿态稳定区：

```text
--keep-pitch-min-x-mm 300
--keep-pitch-max-x-mm 380
```

当 `pitch_source_x_mm` 落在 `[keep_min_x, keep_max_x]` 内时，代码认为固定夹爪姿态已经足够，不做 J2/J3/J5 动态 pitch。

超出稳定区后，只对超出量计算角度：

```text
if x > keep_max_x:
    over_mm = x - keep_max_x
    raw_pitch_delta_deg = over_mm * pitch_deg_per_mm_far

if x < keep_min_x:
    over_mm = x - keep_min_x
    raw_pitch_delta_deg = over_mm * pitch_deg_per_mm_near

if keep_min_x <= x <= keep_max_x:
    over_mm = 0
    raw_pitch_delta_deg = 0
```

业务限幅在关节限位之前执行：

```text
requested_pitch_delta_deg =
    clamp(raw_pitch_delta_deg,
          min_dynamic_pitch_deg,
          max_dynamic_pitch_deg)
```

参数含义：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--adaptive-joint-pitch-by-x` | 关闭 | 开启稳定区外的 J2/J3/J5 关节 pitch 调整。 |
| `--dynamic-pitch-by-x` | 关闭 | 同上，保留旧名称。 |
| `--keep-pitch-min-x-mm` | `300.0` | 固定姿态可抓区域的最小 X。 |
| `--keep-pitch-max-x-mm` | `380.0` | 固定姿态可抓区域的最大 X。 |
| `--pitch-deg-per-mm-far` | `0.04` | 远端超出 keep_max 后，每 1 mm 对应的 pitch 角度。 |
| `--pitch-deg-per-mm-near` | `0.03` | 近端低于 keep_min 后，每 1 mm 对应的 pitch 角度。 |
| `--min-dynamic-pitch-deg` | `-12.0` | 业务允许的最小动态 pitch。 |
| `--max-dynamic-pitch-deg` | `12.0` | 业务允许的最大动态 pitch。 |
| `--dynamic-pitch-speed` | `20` | 关节 pitch 修正使用的速度百分比。 |

旧参数 `--pitch-deg-per-mm` 仍可用；如果提供，它会同时设置 near/far 两侧 slope，用于兼容旧命令。`--pitch-x-ref-mm` 也保留，但新稳定区模型不再依赖它计算角度。

示例：

```text
keep 区间 = 300~380 mm
pitch_deg_per_mm_far = 0.04
pitch_deg_per_mm_near = 0.03

x = 350 mm -> mode=keep_fixed, delta=0
x = 430 mm -> over=50 mm, raw_delta=50 * 0.04 = +2.0 deg
x = 250 mm -> over=-50 mm, raw_delta=-50 * 0.03 = -1.5 deg
```

`pitch_deg_per_mm_far` 和 `pitch_deg_per_mm_near` 是实机标定参数，不是机械臂的固定物理常数。初值可以小一些，然后用抓取成功率、接触状态和日志复盘逐步校准。

## 4. 为什么不是直接改 `ry_deg`

当前实现没有直接修改 `rx/ry/rz` 中的 `ry_deg`，而是在执行固定 waypoint 后调用关节控制做小幅修正。原因是：

- 主流程是固定 J6 笛卡尔位姿序列，直接改 `ry_deg` 会改变 IK、TCP offset 投影和每个 waypoint 的可达性。
- 只在抓取相关阶段叠加关节微调，可以把影响范围限制在抓取动作附近。
- 通过 J2/J3/J5 分配 pitch，必要时可以不只依赖腕部 J5，降低单个关节接近限位的风险。
- 返回的 plan 仍能清楚显示固定笛卡尔姿态，动态变化通过 `joint_pitch_adjustments` 单独记录。

实际链路是：

```text
执行固定 EndPoseCtrl waypoint
读取当前 J2/J3/J5
按比例叠加 pitch_delta
通过 JointCtrl 下发新关节角
```

这会改变真实夹爪姿态，但不会改变 plan 中固定 waypoint 的 `rx/ry/rz` 字段。

## 5. 关节分配方式

动态 pitch 按比例分配到 J2、J3、J5：

```text
j2_delta_deg = pitch_delta_deg * dynamic_pitch_j2_ratio
j3_delta_deg = pitch_delta_deg * dynamic_pitch_j3_ratio
j5_delta_deg = pitch_delta_deg * dynamic_pitch_j5_ratio
```

默认配置：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--dynamic-pitch-j2-ratio` | `0.0` | J2 分担比例。 |
| `--dynamic-pitch-j3-ratio` | `0.0` | J3 分担比例。 |
| `--dynamic-pitch-j5-ratio` | `1.0` | J5 分担比例。 |

也就是默认只改 J5。如果 J5 经常接近限位，可以把 pitch 分散到多个关节：

```bash
--dynamic-pitch-j2-ratio 0.5 \
--dynamic-pitch-j3-ratio -1.0 \
--dynamic-pitch-j5-ratio 0.5
```

ratio 不要求加起来等于 1。它们表示“一个 pitch_delta 分别要给每个关节加多少倍”。如果开启 dynamic pitch 且三个 ratio 都是 0，参数校验会直接报错。

## 6. 执行流程

整体链路：

```text
target_from_command()
  - 手眼变换得到 base_xyz_m
  - 得到 pick_x_mm / pick_y_mm
  - 保存 raw_pick_x_mm / raw_pick_y_mm
  - pitch_source_x_mm = raw_pick_x_mm 或 pick_x_mm

joint_pitch_decision()
  - 判断 keep_fixed / adjust
  - 计算 over_mm
  - 计算 raw_pitch_delta_deg
  - 业务限幅得到 requested_pitch_delta_deg

build_steps()
  - mode=keep_fixed: 不标记 joint_pitch
  - mode=adjust: 对抓取相关 waypoint 标记 joint_pitch_before_pose / joint_pitch_after_pose

execute_step()
  - 执行固定 EndPoseCtrl waypoint
  - 到达后重新计算并施加 J2/J3/J5 joint pitch
  - 再执行夹爪 open / close
```

动态 pitch 只作用在抓取相关步骤，不作用在固定放置、释放和回 ready 的步骤。

`out` 任务中主要作用于：

- `pick_above`
- `pick_grasp`
- `pick_lift`

`in` 任务中主要作用于：

- `return_pick_above`
- `return_pick_grasp`
- `return_pick_lift`

如果启用了 guarded descent，最终下降由保护下降逻辑执行；下降完成后会用 `phase=after_guarded_descent` 再做一次 pitch 修正，然后再闭合夹爪。

## 7. 关节限位保护

在真正下发 `JointCtrl` 前，代码会读取当前关节反馈：

```text
current_joints = controller.get_joint_state()
```

然后根据 J2、J3、J5 的关节限位，计算当前姿态下允许的 pitch delta 区间。实现中留了 `0.1 deg` 余量，避免四舍五入到 SDK 毫度单位后刚好越界。

最终下发角度是：

```text
pitch_delta_deg =
    clamp(requested_pitch_delta_deg, feasible_lower, feasible_upper)
```

返回字段区分三层值：

- `raw_pitch_delta_deg`：按距离和 slope 算出来的原始值。
- `requested_pitch_delta_deg`：经过业务 min/max 限幅后的值。
- `pitch_delta_deg`：经过 J2/J3/J5 关节限位二次夹紧后最终执行的值。
- `clamped=true`：说明 raw 被业务角度限制夹过。
- `limit_clamped=true`：说明 requested 被关节限位夹过。

如果当前关节反馈无效，动态 pitch 会报错，因为没有可靠的当前关节角就不能安全叠加关节增量。

## 8. 调试日志和返回字段

开启 `--debug-plan` 后，会打印 dynamic pitch 摘要：

```text
dynamic_pitch:
  method=joint_space_j2_j3_j5
  mode=adjust
  source_x=430.000mm
  keep=[300.000,380.000]mm
  over=50.000mm
  k_near=0.030000deg/mm
  k_far=0.040000deg/mm
  raw_delta=2.000deg
  requested=2.000deg
  delta=2.000deg
  j2=0.000
  j3=0.000
  j5=1.000
  clamped=False
  limit_clamped=False
```

稳定区内会显示：

```text
mode=keep_fixed
over=0.000mm
raw_delta=0.000deg
delta=0.000deg
```

真实执行时，每次关节修正会打印：

```text
Applying dynamic joint pitch:
  step=pick_grasp
  phase=after_pose
  pick_x=350.000mm
  source_x=430.000mm
  mode=adjust
  over=50.000mm
  pitch_delta=2.000deg
  requested=2.000deg
  j2=0.000deg
  j3=0.000deg
  j5=2.000deg
```

ACK payload 中会包含：

- `dynamic_pitch`：本次目标对应的决策对象，包括 `mode`、`source_x_mm`、`keep_min_x_mm`、`keep_max_x_mm`、`over_mm`、三层 delta、ratio 和 clamp 标志。
- `joint_pitch_adjustments`：每一次实际关节修正的详细记录，包括 `step_label`、`phase`、`start_joints_deg`、`target_joints_deg`、`j2_delta_deg`、`j3_delta_deg`、`j5_delta_deg`。

当 `mode=keep_fixed` 时，不会产生 `joint_pitch_adjustments`。

## 9. 推荐调参步骤

1. 关闭 dynamic pitch，用固定姿态实测找出固定夹爪姿态可稳定抓取的 X 范围，例如 `300~380 mm`。
2. 设置 `--keep-pitch-min-x-mm` 和 `--keep-pitch-max-x-mm`，确认区间内 `mode=keep_fixed`，且没有 `JointCtrl` 调整。
3. 在远端选几个测试点，例如 `400/430/460/490 mm`，分别试小角度 pitch，记录抓取最稳定的角度。
4. 用“推荐角度 / 超出 keep_max 的距离”估算 `--pitch-deg-per-mm-far`。
5. 在近端用同样方法估算 `--pitch-deg-per-mm-near`。近端和远端不要默认共用一个 slope。
6. 初期使用较小业务范围，例如 `--min-dynamic-pitch-deg -6 --max-dynamic-pitch-deg 6`，确认方向正确后再放宽。
7. 如果日志中经常出现 `limit_clamped=true`，说明某个参与关节接近限位。可以降低动态角度，或把 J5 的动作分摊给 J2/J3。

推荐初始命令片段：

```bash
python3 -m robot_control.linux.catch.catch_server_1 \
  --execute \
  --can can0 \
  --hand-eye-result robot_control/linux/hand_eye/hand_eye_result.json \
  --adaptive-joint-pitch-by-x \
  --keep-pitch-min-x-mm 300 \
  --keep-pitch-max-x-mm 380 \
  --pitch-deg-per-mm-far 0.04 \
  --pitch-deg-per-mm-near 0.03 \
  --min-dynamic-pitch-deg -12 \
  --max-dynamic-pitch-deg 12 \
  --dynamic-pitch-j5-ratio 1.0 \
  --dynamic-pitch-speed 20 \
  --debug-plan
```

## 10. 使用边界

- 当前“距离”使用的是机器人基座坐标 X，不是相机深度 `z_m`。如果现场物体远近主要不沿 base X 变化，需要改成使用其他距离量。
- `pitch_deg_per_mm_far` 和 `pitch_deg_per_mm_near` 是经验标定参数，不能假设在所有高度、物体尺寸和夹爪接触状态下都成立。
- 动态 pitch 是关节空间增量，不是完整的重新规划。角度应该保持小范围，适合作为抓取姿态补偿。
- `force_pick_x_mm` 只改变最终抓取 X；动态 pitch 默认仍使用 `raw_pick_x_mm`，这是为了在固定抓取线场景里保留真实距离信息。
- 若 `command_duration_s` 很短，`JointCtrl` 发送次数会减少，实际姿态变化可能不明显。
- 如果夹爪姿态变化后导致 TCP 位置偏移明显，需要重新检查抓取高度、保护下降和夹爪闭合时机。
