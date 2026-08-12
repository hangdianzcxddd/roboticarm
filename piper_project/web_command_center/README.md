# Piper Command Center

这个子项目提供一个纯 Linux 单机 Web 控制台：

- 启动同一台机器上的视觉检测和机械臂抓取服务。
- 浏览器显示 OpenCV RGB、mask、depth 预览。
- 浏览器显示 Linux 重要日志和完整输出。

启动：

```bash
python3 -m web_command_center.server --host 127.0.0.1 --port 8088
```

打开：

```text
http://127.0.0.1:8088
```

当前 Web 默认按现场实测参数生成：视觉使用
`robot_control.windows.client.opencv_grasp_send_tcp`，抓取服务使用
`robot_control.linux.catch.catch_server_1_1`。新版运行配置指向
当前项目目录，命令直接在本机执行，不再使用 SSH、Windows 虚拟机或远程路径。

Web 服务启动视觉命令时会自动设置 `PIPER_WEB_PREVIEW_DIR`，OpenCV 客户端会把
检测画面写入 `web_command_center/runtime/preview/`，页面会自动刷新这些 JPG 文件。

注意：这个页面会执行生成的本机命令，默认只监听 `127.0.0.1`。需要局域网访问时再改
`--host 0.0.0.0`。

## 迁移验收顺序

1. 激活 `piper` 环境，确认 `python -c "import cv2, pyrealsense2, can"` 成功；Piper SDK 使用项目相邻目录 `sdk/piper_sdk`，由机械臂控制模块自动加入路径。
2. 插入 RealSense 后运行 `rs-enumerate-devices`。若 `pyrealsense2` 报 udev 权限错误，安装 librealsense 的 udev 规则或把用户加入 `plugdev`，重新插拔相机；不要求必须使用 Debian 软件包，pip/conda 的 `pyrealsense2` 可以完成视觉脚本功能。
3. 检查 CAN 适配器是否出现：`ip link show`。首次配置可在页面输入 sudo 密码，或执行 `sudo bash ./setup_can.sh can0 1000000`；脚本必须用 Bash 和 LF 换行。
4. 启动 Web 服务后点击“检查 5005”，必须显示本机端口空闲；不要再填写虚拟机 IP、SSH 用户或远程项目路径。
5. 先启动抓取服务。无 CAN 时可以暂不勾选真实执行，仅验证服务监听；接入机械臂后再使用当前配置中的 `--execute`。
6. 启动视觉检测，确认 RGB/mask/depth 预览更新，并在抓取服务日志中看到客户端连接和 JSON Lines `camera_point` 应答。
7. 最后在低速、空载条件下验证准备位、抓取位、抬升和释放动作，再恢复传送带和全部动态参数。

视觉到机械臂的数据链路是单机 TCP：视觉进程连接 `127.0.0.1:5005`，每行发送一个 JSON 对象，抓取服务返回一行 `ack` 或 `error`。不再经过 SSH、Windows 主机、虚拟机网卡或远程端口转发。
