const state = {
  profile: null,
  commands: { vision: "", robot: "" },
  activePreview: "rgb",
  previewReady: false,
  streamConnected: false,
  activeCommand: "windowsCommand",
  activeLog: "all",
};

const el = {
  profileName: document.getElementById("profileName"),
  linuxSudoPassword: document.getElementById("linuxSudoPassword"),
  windowsCommand: document.getElementById("windowsCommand"),
  linuxCommand: document.getElementById("linuxCommand"),
  windowsLog: document.getElementById("windowsLog"),
  linuxLog: document.getElementById("linuxLog"),
  allLog: document.getElementById("allLog"),
  linuxImportantLog: document.getElementById("linuxImportantLog"),
  windowsStatus: document.getElementById("windowsStatus"),
  linuxStatus: document.getElementById("linuxStatus"),
  linuxPortStatus: document.getElementById("linuxPortStatus"),
  lastLinuxEvent: document.getElementById("lastLinuxEvent"),
  validationStatus: document.getElementById("validationStatus"),
  startWindows: document.getElementById("startWindows"),
  startLinux: document.getElementById("startLinux"),
  armActivityBar: document.getElementById("armActivityBar"),
  armActivityStep: document.getElementById("armActivityStep"),
  armActivityPhase: document.getElementById("armActivityPhase"),
  armActivityTime: document.getElementById("armActivityTime"),
  armActivityHistory: document.getElementById("armActivityHistory"),
  previewImage: document.getElementById("previewImage"),
  previewEmpty: document.getElementById("previewEmpty"),
  pauseLogs: document.getElementById("pauseLogs"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || payload.message || response.statusText);
  }
  return payload;
}

async function loadProfile() {
  const payload = await api("/api/profile");
  state.profile = payload.profile;
  state.commands = payload.commands;
  bindProfileToForm();
  renderCommands();
  renderValidation(payload.validation);
}

function bindProfileToForm() {
  if (!state.profile) return;
  el.profileName.value = state.profile.name || "";
  document.querySelectorAll("[data-path]").forEach((input) => {
    const value = getPath(state.profile, input.dataset.path);
    if (input.type === "checkbox") {
      input.checked = Boolean(value);
    } else {
      input.value = value ?? "";
    }
  });
}

function updateProfileFromForm() {
  if (!state.profile) return;
  state.profile.name = el.profileName.value.trim() || "未命名配置";
  document.querySelectorAll("[data-path]").forEach((input) => {
    let value;
    if (input.type === "checkbox") {
      value = input.checked;
    } else if (input.type === "number") {
      value = input.value === "" ? null : Number(input.value);
    } else {
      value = input.value;
    }
    setPath(state.profile, input.dataset.path, value);
  });
}

function getPath(source, path) {
  return path.split(".").reduce((current, part) => {
    if (current == null) return undefined;
    return current[part];
  }, source);
}

function setPath(source, path, value) {
  const parts = path.split(".");
  let current = source;
  for (let index = 0; index < parts.length - 1; index += 1) {
    const part = parts[index];
    const nextPart = parts[index + 1];
    if (current[part] == null) {
      current[part] = /^\d+$/.test(nextPart) ? [] : {};
    }
    current = current[part];
  }
  current[parts[parts.length - 1]] = value;
}

async function previewProfile() {
  updateProfileFromForm();
  const validation = await api("/api/profile/validate", {
    method: "POST",
    body: JSON.stringify({ profile: state.profile }),
  });
  state.commands = validation.commands;
  renderCommands();
  renderValidation(validation);
  return validation;
}

async function saveProfile() {
  updateProfileFromForm();
  const payload = await api("/api/profile", {
    method: "POST",
    body: JSON.stringify({ profile: state.profile }),
  });
  state.profile = payload.profile;
  state.commands = payload.commands;
  renderCommands();
  renderValidation(payload.validation);
}

async function startProcess(name) {
  const validation = await previewProfile();
  if (!validation.ok) {
    throw new Error(validation.errors.join("\n"));
  }
  await api(`/api/process/${name}/start`, {
    method: "POST",
    body: JSON.stringify({ profile: state.profile }),
  });
  await refreshStatus();
}

async function stopProcess(name) {
  await api(`/api/process/${name}/stop`, { method: "POST", body: "{}" });
  await refreshStatus();
}

async function stopAll() {
  await Promise.all([stopProcess("vision"), stopProcess("robot")]);
}

async function toggleProcess(name) {
  const button = name === "vision" ? el.startWindows : el.startLinux;
  if (button.dataset.running === "true") {
    await stopProcess(name);
  } else {
    await startProcess(name);
  }
}

async function checkLinuxPort() {
  updateProfileFromForm();
  const status = await api("/api/robot/port-status", {
    method: "POST",
    body: JSON.stringify({ profile: state.profile }),
  });
  renderLinuxPortStatus(status);
  return status;
}

async function stopRemoteLinuxPortOwner() {
  updateProfileFromForm();
  const status = await checkLinuxPort();
  if (!status.ok) {
    throw new Error(status.message || "本机端口检查失败");
  }
  if (!status.occupied) {
    alert(status.message);
    return status;
  }
  const detail = status.pids && status.pids.length ? ` PID ${status.pids.join(", ")}` : "";
  const ok = window.confirm(`本机端口 ${status.port} 已被旧服务占用${detail}。\n确定要停止它吗？`);
  if (!ok) return status;
  const result = await api("/api/robot/stop-port-owner", {
    method: "POST",
    body: JSON.stringify({ profile: state.profile }),
  });
  renderLinuxPortStatus(result.status || result);
  alert(result.message);
  return result;
}

async function setupCan() {
  updateProfileFromForm();
  const password = el.linuxSudoPassword.value;
  if (!password) {
    throw new Error("请输入 Linux sudo 密码");
  }
  const result = await api("/api/robot/setup-can", {
    method: "POST",
    body: JSON.stringify({ profile: state.profile, password }),
  });
  el.linuxSudoPassword.value = "";
  alert(result.message);
  await refreshLogs();
  await refreshStatus();
  return result;
}

function renderCommands() {
  el.windowsCommand.textContent = state.commands.vision || state.commands.windows || "";
  el.linuxCommand.textContent = state.commands.robot || state.commands.linux || "";
}

function renderValidation(validation) {
  el.validationStatus.classList.remove("running", "error", "warn");
  if (!validation) {
    el.validationStatus.textContent = "配置未校验";
    return;
  }
  if (!validation.ok) {
    el.validationStatus.classList.add("error");
    el.validationStatus.textContent = `配置错误 ${validation.errors.length}`;
    return;
  }
  if (validation.warnings && validation.warnings.length) {
    el.validationStatus.classList.add("warn");
    el.validationStatus.textContent = validation.warnings.join(" / ");
    return;
  }
  el.validationStatus.classList.add("running");
  el.validationStatus.textContent = "配置可运行";
}

function renderLinuxPortStatus(status) {
  el.linuxPortStatus.classList.remove("running", "error", "warn");
  if (!status) {
    el.linuxPortStatus.textContent = "5005 未检查";
    return;
  }
  if (!status.ok) {
    el.linuxPortStatus.classList.add("error");
    el.linuxPortStatus.textContent = status.message || "本机端口检查失败";
    return;
  }
  if (status.occupied) {
    el.linuxPortStatus.classList.add("warn");
    const pidText = status.pids && status.pids.length ? ` PID ${status.pids.join(",")}` : "";
    el.linuxPortStatus.textContent = `本机 ${status.port} 已占用${pidText}`;
    return;
  }
  el.linuxPortStatus.classList.add("running");
  el.linuxPortStatus.textContent = `本机 ${status.port} 空闲`;
}

function renderStatus(name, status) {
  const target = name === "vision" ? el.windowsStatus : el.linuxStatus;
  const button = name === "vision" ? el.startWindows : el.startLinux;
  target.classList.toggle("running", Boolean(status.running));
  target.classList.toggle("error", !status.running && status.exit_code && status.exit_code !== 0);
  button.dataset.running = status.running ? "true" : "false";
  button.classList.toggle("primary", !status.running);
  button.classList.toggle("danger", Boolean(status.running));
  button.textContent = status.running
    ? `停止 ${name === "vision" ? "视觉" : "抓取服务"}`
    : `启动 ${name === "vision" ? "视觉" : "抓取服务"}`;
  const prefix = name === "vision" ? "视觉" : "抓取服务";
  if (status.running) {
    const uptime = status.uptime_s == null ? "" : ` ${Math.round(status.uptime_s)}s`;
    target.textContent = `${prefix} running${uptime}`;
  } else if (status.exit_code != null) {
    target.textContent = `${prefix} exited ${status.exit_code}`;
  } else {
    target.textContent = `${prefix} idle`;
  }
}

function renderLastLinuxEvent(status) {
  const item = status.robot.last_important;
  if (!item) {
    el.lastLinuxEvent.textContent = "无关键事件";
    return;
  }
  el.lastLinuxEvent.textContent = `${item.time} ${item.level}`;
}

function renderArmActivity(activity) {
  if (!activity || !el.armActivityBar) return;
  el.armActivityBar.classList.remove("running", "done", "idle", "error");
  const phase = activity.phase || "空闲";
  const label = activity.label || "等待机械臂流程日志";
  const time = activity.time ? `[${activity.time}]` : "";
  if (activity.running) {
    el.armActivityBar.classList.add("running");
  } else if (activity.exit_code && activity.exit_code !== 0) {
    el.armActivityBar.classList.add("error");
  } else if (phase === "完成") {
    el.armActivityBar.classList.add("done");
  } else {
    el.armActivityBar.classList.add("idle");
  }
  el.armActivityStep.textContent = label;
  el.armActivityPhase.textContent = phase;
  el.armActivityTime.textContent = time;
}

function renderArmActivityHistory(history) {
  if (!el.armActivityHistory) return;
  const items = Array.isArray(history) ? history.slice(-18).reverse() : [];
  if (!items.length) {
    el.armActivityHistory.innerHTML = '<div class="activity-history-empty">暂无状态记录</div>';
    return;
  }
  el.armActivityHistory.innerHTML = items
    .map((item) => {
      const time = item.time ? `[${escapeHtml(item.time)}]` : "";
      const label = escapeHtml(item.label || item.step || "未知状态");
      const phase = escapeHtml(item.phase || "");
      return (
        '<div class="activity-history-item">' +
        `<span class="activity-history-time">${time}</span>` +
        '<div class="activity-history-main">' +
        `<div class="activity-history-label">${label}</div>` +
        `<div class="activity-history-phase">${phase}</div>` +
        "</div>" +
        "</div>"
      );
    })
    .join("");
}

function labelArmStep(step) {
  const labels = {
    initial_ready: "初始准备等待",
    startup_ready: "启动准备位",
    ready_open: "准备并打开夹爪",
    pick_above: "移动到抓取上方",
    preposition: "预定位等待",
    dynamic_preposition: "动态预定位",
    pick_grasp: "下探抓取",
    pick_lift: "抓取后抬升",
    return_ready_above: "返回准备上方",
    place_above: "移动到放置上方",
    place_release: "放置并释放",
    return_ready: "返回准备位",
  };
  return labels[step] || String(step || "").replaceAll("_", " ");
}

function parseArmActivityLog(line) {
  const text = line.text || "";
  const stepMatch = text.match(/\bstep=([a-zA-Z0-9_-]+)/);
  if ((text.includes("Running planar IK") || text.includes("Reusing planar IK")) && stepMatch) {
    const step = stepMatch[1];
    return { phase: "运动中", step, label: labelArmStep(step), time: line.time };
  }
  if (text.includes("Planar IK") && text.includes("feedback:") && stepMatch) {
    const step = stepMatch[1];
    return { phase: "到达反馈", step, label: labelArmStep(step), time: line.time };
  }
  const runningMatch = text.match(/\bRunning ([a-zA-Z0-9_-]+):/);
  if (runningMatch) {
    const step = runningMatch[1];
    return { phase: "运动中", step, label: labelArmStep(step), time: line.time };
  }
  const waitMatch = text.match(/Dynamic final timing: waiting [0-9.]+s at ([^(]+)/);
  if (waitMatch) {
    const step = waitMatch[1].trim().replaceAll(" ", "_");
    return { phase: "动态时序等待", step, label: labelArmStep(step), time: line.time };
  }
  if (text.includes("Client connected:")) {
    return { phase: "连接", step: "client_connected", label: "客户端已连接", time: line.time };
  }
  if (text.includes("Before catch J6")) {
    return { phase: "抓取开始", step: "before_catch", label: "读取抓取前姿态", time: line.time };
  }
  if (text.includes("Moving to startup ready pose")) {
    return { phase: "启动", step: "startup_ready", label: labelArmStep("startup_ready"), time: line.time };
  }
  if (text.includes("Piper arm enabled")) {
    return { phase: "就绪", step: "enabled", label: "机械臂已使能", time: line.time };
  }
  if (text.toLowerCase().includes("listening on")) {
    return { phase: "等待", step: "listening", label: "等待视觉客户端连接", time: line.time };
  }
  if (text.includes("After catch J6")) {
    return { phase: "完成", step: "after_catch", label: "本轮抓取完成", time: line.time };
  }
  if (text.includes("Catch send counts")) {
    return { phase: "完成", step: "catch_counts", label: "动作指令统计完成", time: line.time };
  }
  if (text.startsWith("task=")) {
    return { phase: "完成", step: "task_result", label: "任务结果已输出", time: line.time };
  }
  return null;
}

function renderArmActivityHistoryFromLogs(lines) {
  const history = [];
  let lastKey = "";
  lines.forEach((line) => {
    const item = parseArmActivityLog(line);
    if (!item) return;
    const key = `${item.phase}|${item.step}|${item.label}`;
    if (key === lastKey) return;
    history.push(item);
    lastKey = key;
  });
  renderArmActivityHistory(history);
}

async function refreshStatus() {
  const status = await api("/api/status");
  renderStatus("vision", status.vision);
  renderStatus("robot", status.robot);
  renderLastLinuxEvent(status);
  renderArmActivity(status.arm_activity);
  renderArmActivityHistory(status.arm_activity_history);
}

async function refreshLogs() {
  if (el.pauseLogs.checked) return;
  const [windows, linux, linuxImportant] = await Promise.all([
    api("/api/logs/vision?limit=500"),
    api("/api/logs/robot?limit=500"),
    api("/api/logs/robot?important=1&limit=160"),
  ]);
  renderPlainLog(el.windowsLog, windows);
  renderPlainLog(el.linuxLog, linux);
  renderAllLog(windows, linux);
  renderImportantLog(linuxImportant);
  renderArmActivityHistoryFromLogs(linux);
}

function renderAllLog(windows, linux) {
  const merged = [
    ...windows.map((line) => ({ ...line, source: "VISION" })),
    ...linux.map((line) => ({ ...line, source: "ROBOT" })),
  ].slice(-1000);
  const text = merged.map((line) => `[${line.time}] ${line.source} ${line.text}`).join("\n");
  renderTextLog(el.allLog, text);
}

function renderPlainLog(target, lines) {
  const text = lines.map((line) => `[${line.time}] ${line.text}`).join("\n");
  renderTextLog(target, text);
}

function renderTextLog(target, text) {
  if (target.textContent !== text) {
    const shouldStick = target.scrollTop + target.clientHeight >= target.scrollHeight - 20;
    target.textContent = text;
    if (shouldStick) target.scrollTop = target.scrollHeight;
  }
}

function renderImportantLog(lines) {
  if (!lines.length) {
    el.linuxImportantLog.innerHTML = '<div class="empty">等待抓取服务关键事件</div>';
    return;
  }
  el.linuxImportantLog.innerHTML = lines
    .map(
      (line) =>
        `<div class="log-line ${escapeHtml(line.level)}">` +
        `<span class="time">${escapeHtml(line.time)}</span>` +
        `<span>${escapeHtml(line.text)}</span>` +
        `</div>`,
    )
    .join("");
  el.linuxImportantLog.scrollTop = el.linuxImportantLog.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function refreshPreview() {
  const status = await api("/api/preview/status");
  const current = status[state.activePreview];
  if (!current || !current.exists) {
    state.previewReady = false;
    state.streamConnected = false;
    el.previewImage.classList.remove("ready");
    el.previewEmpty.style.display = "block";
    return;
  }
  if (!state.streamConnected) {
    el.previewImage.src = `/stream/${state.activePreview}.mjpg?t=${Date.now()}`;
    state.streamConnected = true;
  }
}

function selectPage(pageId) {
  document.querySelectorAll(".page-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === pageId);
  });
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.id === pageId);
  });
}

function selectPreview(preview) {
  document.querySelectorAll(".preview-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.preview === preview);
  });
  state.activePreview = preview;
  state.previewReady = false;
  state.streamConnected = false;
  el.previewImage.classList.remove("ready");
  el.previewEmpty.style.display = "block";
  el.previewImage.removeAttribute("src");
  refreshPreview().catch(console.error);
}

function selectCommand(commandId) {
  state.activeCommand = commandId;
  document.querySelectorAll(".command-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.command === commandId);
  });
  document.querySelectorAll(".generated-command").forEach((target) => {
    target.classList.toggle("hidden", target.id !== commandId);
  });
}

function selectLog(filter) {
  state.activeLog = filter;
  document.querySelectorAll(".log-filter").forEach((button) => {
    button.classList.toggle("active", button.dataset.filter === filter);
  });
  document.querySelectorAll("[data-log-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.logPanel !== filter);
  });
}

function alertError(error) {
  alert(error.message || String(error));
}

function debounce(fn, waitMs) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), waitMs);
  };
}

const debouncedPreviewProfile = debounce(() => previewProfile().catch(console.error), 250);

document.querySelectorAll("[data-path]").forEach((input) => {
  input.addEventListener("input", debouncedPreviewProfile);
  input.addEventListener("change", debouncedPreviewProfile);
});

el.profileName.addEventListener("input", debouncedPreviewProfile);
el.previewImage.addEventListener("load", () => {
  state.previewReady = true;
  el.previewImage.classList.add("ready");
  el.previewEmpty.style.display = "none";
});

document.querySelectorAll(".page-tab").forEach((button) => {
  button.addEventListener("click", () => selectPage(button.dataset.page));
});

document.querySelectorAll(".preview-tab").forEach((button) => {
  button.addEventListener("click", () => selectPreview(button.dataset.preview));
});

document.querySelectorAll(".command-tab").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    selectCommand(button.dataset.command);
  });
});

document.querySelectorAll(".log-filter").forEach((button) => {
  button.addEventListener("click", () => selectLog(button.dataset.filter));
});

document.getElementById("saveProfile").addEventListener("click", () =>
  saveProfile().catch(alertError),
);
document.getElementById("validateProfile").addEventListener("click", () =>
  previewProfile().catch(alertError),
);
document.getElementById("checkLinuxPort").addEventListener("click", () =>
  checkLinuxPort().catch(alertError),
);
document.getElementById("setupCan").addEventListener("click", () =>
  setupCan().catch(alertError),
);
document.getElementById("stopRemoteLinux").addEventListener("click", () =>
  stopRemoteLinuxPortOwner().catch(alertError),
);
el.startWindows.addEventListener("click", () => toggleProcess("vision").catch(alertError));
el.startLinux.addEventListener("click", () => toggleProcess("robot").catch(alertError));
document.getElementById("stopAll").addEventListener("click", () => stopAll().catch(alertError));

async function tick() {
  try {
    await Promise.all([refreshStatus(), refreshLogs(), refreshPreview()]);
  } catch (error) {
    console.error(error);
  }
}

loadProfile()
  .then(tick)
  .catch((error) => {
    console.error(error);
    alertError(error);
  });

setInterval(tick, 1000);
