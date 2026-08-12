# Piper Command Center

这个子项目提供一个最小 Web 控制台：

- 一个输入框启动 Windows/OpenCV 检测命令。
- 一个输入框启动 Linux/catch server 命令。
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

建议把这个 Web 服务运行在 Windows 视觉端，因为 OpenCV 检测画面来自
`opencv_grasp_send_tcp.py` 本机进程。Linux 命令框可以直接粘贴本机命令，也可以粘贴
SSH 命令，例如：

```bash
ssh user@192.168.146.128 "cd /home/zcxddd/projects/roboticarm/piper_project && python3 -m robot_control.linux.catch.catch_server_1_1 ..."
```

当前 Web 默认按现场实测参数生成：Windows 使用
`robot_control.windows.client.opencv_grasp_send_tcp`，Linux 使用
`robot_control.linux.catch.catch_server_1_1`。新版运行配置指向
`/home/zcxddd/projects/roboticarm/piper_project`；旧版发布目录
`D:\git\projects\roboticarm\robot_releases\piper_web_v1` 保留为独立回滚服务，
不要把新版 `catch_server_1_1` 命令发送到该旧目录。

Web 服务启动 Windows 命令时会自动设置 `PIPER_WEB_PREVIEW_DIR`，OpenCV 客户端会把
检测画面写入 `web_command_center/runtime/preview/`，页面会自动刷新这些 JPG 文件。

注意：这个页面会执行输入框里的任意命令，默认只监听 `127.0.0.1`。需要局域网访问时再改
`--host 0.0.0.0`。
