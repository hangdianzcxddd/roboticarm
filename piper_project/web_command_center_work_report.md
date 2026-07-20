# Web Command Center 工作汇报

日期：2026-07-13  
项目目录：`D:\git\projects\roboticarm\piper_project`  
模块：`web_command_center`

## 一、工作背景

本次工作围绕 `web_command_center` 进行改造，目标是把原本依赖长 Shell 命令启动的机械臂抓取流程，整理为一个更适合现场操作的 Web 控制界面。改造重点包括：参数可视化配置、Windows 视觉端与 Linux 机械臂端分区管理、日志与画面分离、机械臂状态展示、CAN 连接能力集成、服务启停控制，以及固定版本运行方案。

## 二、主要目标

1. 将 Windows 视觉端和 Linux 机械臂端的重要参数改为 Web 表单配置。
2. 隐藏固定基础连接参数，降低误操作概率。
3. 将实时画面展示、命令配置、日志查看拆分为更清晰的界面结构。
4. 增加机械臂当前流程状态和历史状态记录。
5. 增加通过 `setup_can.sh` 连接机械臂 CAN 的 Web 操作入口。
6. 支持 Windows 和 Linux 服务的启动、停止、状态同步。
7. 形成固定 Git 提交版本运行方案，避免后续后端代码更新影响当前稳定版本。

## 三、界面改造内容

### 1. 参数区重新分栏

界面被整理为三列：

- 左列：Windows 视觉参数。
- 中列：Linux 运动参数。
- 右列：画面展示与机械臂状态。

Linux 运动参数已移动到第二列，并放在抓取目标参数上方。两列参数字体和间距缩小，使参数区更紧凑，右侧画面展示区域获得更大的显示空间。

### 2. 参数模块命名优化

对原有较偏底层的名称做了更直观的解释和调整建议：

- `ROI 和阈值` 更适合命名为 `画面检测与颜色过滤`。
- 动态窗口用于机械臂抓取范围，视觉 ROI 用于图像裁剪检测，两者需要区分。
- `Close kernel` 可理解为 `填补空洞强度`。
- `Open kernel` 可理解为 `去除噪声强度`。

其中闭运算主要用于补全检测目标内部缺口，开运算主要用于去掉小面积噪点。

### 3. 参数面板滚动

为参数设置区域增加滚动能力，解决部分模块内容超出容器后无法显示和无法设置的问题。尤其是 `Linux 高级` 等参数较多的区域，可以通过滚动完整访问。

### 4. 生成命令区域调整

生成命令模块已移动到 Windows 视觉和 Linux 运动两栏的下方，默认折叠，不再打开页面时直接显示长命令。需要查看时手动展开。

## 四、画面与状态展示改造

### 1. 画面展示区

右侧保留视觉画面展示区域，用于显示 OpenCV 输出的预览图，包括 RGB、mask、depth 等运行画面。

### 2. 当前机械臂状态行

在画面展示上方增加机械臂当前状态行，用于从 Linux 日志中提取当前流程，不展示冗长参数，只展示机械臂正在执行的步骤。例如：

- `ready_open`
- `pick_above`
- `dynamic_preposition`
- `pick_grasp`
- `pick_lift`
- `return_ready_above`
- `place_above`
- `place_release`
- `return_ready`

后端通过解析 Linux 服务日志中的关键字段，例如 `step=...` 或 `Running ...`，生成结构化状态。

### 3. 机械臂状态历史

机械臂状态历史记录放在右侧画面展示模块下方，作为画面展示区的一部分。高度调整为 `70px`，并增加横向和纵向滚动，避免长状态文本挤占画面展示区域。

## 五、日志页面改造

日志从主操作页面中拆分到独立页面或独立区域，通过点击切换方式查看。这样主页面保留命令输入、参数配置和画面展示，日志页面用于排查运行问题。

日志区保留 Windows 视觉端和 Linux 机械臂端输出，便于定位：

- SSH 连接问题。
- Linux 服务端口占用。
- Python 环境缺失依赖。
- 机械臂运行流程异常。
- OpenCV 检测和发送任务异常。

## 六、后端功能改造

### 1. 默认命令结构化生成

后端将原始长命令拆为结构化 profile 配置，再根据配置生成命令。

Windows 视觉端保留核心参数：

- 传送带速度。
- 动态抓取 X/Y 范围。
- 画面宽高。
- HSV 颜色过滤。
- 高亮剔除。
- 面积和尺寸阈值。
- 视觉 ROI。
- 入场边缘判断。
- 形态学处理参数。
- 手眼标定结果文件。

Linux 机械臂端保留核心参数：

- 机械臂速度。
- 姿态角。
- 命令持续时间。
- 夹爪稳定时间。
- 动态预定位。
- 抓取高度。
- 检测高度补偿。
- Planar IK 参数。
- 动态时序参数。
- 抓取 yaw 参数。
- 执行开关。

其中 `--command-duration-s 0.3` 和 `--gripper-settle-s 0.1` 已确认保留，没有遗漏。

### 2. 隐藏固定连接参数

以下参数作为默认值处理，不在界面中作为常规参数暴露：

- Windows 连接 Linux 的目标主机。
- Linux 服务监听地址。
- SSH 用户、主机、保活参数。
- Linux 项目路径。
- Conda 初始化脚本。
- Conda 环境名。

这样可以减少现场操作时误改基础连接配置的风险。

### 3. 移除 descent-protection 配置

根据实际运行需求，已经去掉 `--no-descent-protection` 相关内容，不再在新服务中生成或展示。

### 4. Windows 端默认使用 piper 环境

Windows 视觉命令已改为通过 Conda 环境运行：

```powershell
conda run -n piper python -m robot_control.windows.client.opencv_grasp_send_tcp ...
```

解决了直接运行 `python` 时缺少 `cv2` 的问题。

### 5. Linux 端口占用检查

针对 Linux 报错：

```text
OSError: [Errno 98] Address already in use
```

增加了 Linux 服务端口状态检查和释放能力：

- `/api/linux/port-status`
- `/api/linux/stop-port-owner`

用于发现并停止占用 `5005` 端口的旧进程，避免每次重启 Linux 服务时端口冲突。

### 6. setup_can.sh 集成

新增了 Web 中连接机械臂 CAN 的能力。界面只需要输入 Linux sudo 密码，然后后端通过 SSH 在远端项目目录执行：

```bash
bash ./setup_can.sh can0 1000000
```

密码只通过请求体和标准输入传递，不写入配置文件和日志。

后端接口：

```text
POST /api/linux/setup-can
```

### 7. 启动按钮改为启停切换

Windows 和 Linux 启动按钮改为状态切换：

- 未运行时显示 `启动 Windows` / `启动 Linux`。
- 运行后显示 `停止 Windows` / `停止 Linux`。

按钮状态根据 `/api/status` 返回的进程状态同步。

## 七、固定版本运行方案

为避免后续后端代码更新影响当前稳定 Web 服务，采用 Git worktree 固定提交版本。

固定提交：

```text
ff07522e446a206e354794e52477cae0d78372d4
```

提交说明：

```text
web服务第一版完成
```

Windows 固定目录：

```text
D:\git\projects\roboticarm\robot_releases\piper_web_v1\piper_project
```

Linux 固定目录：

```text
/home/zcxddd/projects/roboticarm/releases/piper_web_v1
```

Web 服务启动命令：

```powershell
cd D:\git\projects\roboticarm\robot_releases\piper_web_v1\piper_project
conda activate piper
python -m web_command_center.server --host 127.0.0.1 --port 8088
```

启动时应确认输出：

```text
Project root: D:\git\projects\roboticarm\robot_releases\piper_web_v1\piper_project
```

Web 配置文件：

```text
D:\git\projects\roboticarm\robot_releases\piper_web_v1\piper_project\web_command_center\state\profile.json
```

关键配置：

```json
{
  "linux_shell": {
    "remote_project": "/home/zcxddd/projects/roboticarm/releases/piper_web_v1"
  }
}
```

该配置必须使用无 BOM UTF-8。之前出现过 PowerShell 写入 UTF-8 BOM，导致 Python `json.loads` 解析失败，并回退到默认路径的问题。该问题已通过重写无 BOM UTF-8 配置解决。

## 八、问题排查记录

### 1. SSH BatchMode 权限问题

现象：

```text
ssh: connect to host 192.168.146.128 port 22: Permission denied
```

原因分析：

- 手动 SSH 可以登录，说明网络和账号可用。
- Web 中使用 `BatchMode=yes`，不会弹出交互式密码输入。
- 需要配置 SSH key，或去掉 BatchMode 并支持密码输入。

当前方案：

- 后续服务保持默认 SSH 配置。
- Linux sudo 密码只用于 `setup_can.sh`，不用于普通 SSH 登录。

### 2. Linux 服务端口占用

现象：

```text
OSError: [Errno 98] Address already in use
```

原因：

- 旧的 `catch_server_1` 仍在监听 `5005`。

处理：

- 增加端口检测和停止端口占用进程能力。

### 3. Windows 缺少 cv2

现象：

```text
ModuleNotFoundError: No module named 'cv2'
```

原因：

- Web 服务默认 Python 环境不是 `piper`。

处理：

- Windows 视觉命令改为 `conda run -n piper python ...`。

### 4. 固定目录配置未生效

现象：

生成的 Linux 命令仍然是：

```bash
cd /home/zcxddd/projects/roboticarm/piper_project
```

原因：

- `profile.json` 文件带 UTF-8 BOM。
- Python 使用 `encoding="utf-8"` 读取后，`json.loads` 报错并回退默认配置。

处理：

- 将 `profile.json` 重写为无 BOM UTF-8。
- 重启 Web 服务后，生成命令正确变为：

```bash
cd /home/zcxddd/projects/roboticarm/releases/piper_web_v1
```

## 九、当前成功标志

当前生成的 Linux 命令已经包含固定目录：

```bash
cd /home/zcxddd/projects/roboticarm/releases/piper_web_v1
```

这说明 Linux 服务启动路径已经指向固定版本目录。

完整成功需要同时满足：

1. Windows Web 服务启动输出：

```text
Project root: D:\git\projects\roboticarm\robot_releases\piper_web_v1\piper_project
```

2. Linux 远程目录版本：

```powershell
ssh zcxddd@192.168.146.128 "git -C /home/zcxddd/projects/roboticarm/releases/piper_web_v1 rev-parse HEAD"
```

输出：

```text
ff07522e446a206e354794e52477cae0d78372d4
```

3. Web 配置接口：

```text
http://127.0.0.1:8088/api/profile
```

返回：

```json
"remote_project": "/home/zcxddd/projects/roboticarm/releases/piper_web_v1"
```

## 十、涉及文件

主要涉及文件：

```text
web_command_center/server.py
web_command_center/static/index.html
web_command_center/static/app.js
web_command_center/static/styles.css
setup_can.sh
web_command_center/state/profile.json
```

其中 `web_command_center/state/profile.json` 属于运行配置，不应把密码等敏感内容写入该文件。

## 十一、后续建议

1. 将固定版本目录作为生产运行目录，日常开发继续在主项目目录进行。
2. 不要在 `docker-compose.yml` 中使用 `.:/app` 覆盖镜像代码，否则固定版本会失效。
3. 如果继续使用 Docker，建议只固定 Web 服务代码，Windows 视觉端仍在 Windows 主机 `piper` 环境运行。
4. 如需进一步稳定运行环境，应冻结 Windows 和 Linux 两端的 Conda 环境依赖。
5. 建议增加版本展示接口，在 Web 页面显示当前 `PROJECT_ROOT`、Git commit 和远程 `remote_project`，便于现场确认是否运行在固定版本。
6. 建议在后续版本中让 `load_profile()` 支持 `utf-8-sig`，避免 Windows 工具写入 BOM 后配置静默失效。

## 十二、结论

本次工作已经将 `web_command_center` 从长命令手动启动方式，改造成以 Web 表单、状态展示、日志切换、进程控制和固定版本运行为核心的现场操作工具。当前关键链路已经打通：

- Windows Web 服务可从固定 release 目录启动。
- Windows 视觉命令默认使用 `piper` Conda 环境。
- Linux 服务命令已指向固定 worktree 目录。
- CAN 连接可通过 Web 输入 sudo 密码触发。
- 机械臂状态可从日志中提取并展示。
- 日志、画面、参数配置的界面职责已拆分。

后续重点应放在依赖环境冻结、版本信息显式展示和现场异常提示优化上。
