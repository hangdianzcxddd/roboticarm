const state = {
  activePreview: "rgb",
  previewReady: false,
  streamConnected: false,
};

const el = {
  windowsCommand: document.getElementById("windowsCommand"),
  linuxCommand: document.getElementById("linuxCommand"),
  windowsLog: document.getElementById("windowsLog"),
  linuxLog: document.getElementById("linuxLog"),
  linuxImportantLog: document.getElementById("linuxImportantLog"),
  windowsStatus: document.getElementById("windowsStatus"),
  linuxStatus: document.getElementById("linuxStatus"),
  lastLinuxEvent: document.getElementById("lastLinuxEvent"),
  previewImage: document.getElementById("previewImage"),
  previewEmpty: document.getElementById("previewEmpty"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

async function loadCommands() {
  const commands = await api("/api/commands");
  el.windowsCommand.value = commands.windows || "";
  el.linuxCommand.value = commands.linux || "";
}

async function saveCommands() {
  await api("/api/commands", {
    method: "POST",
    body: JSON.stringify({
      windows: el.windowsCommand.value,
      linux: el.linuxCommand.value,
    }),
  });
}

async function startProcess(name) {
  const command = name === "windows" ? el.windowsCommand.value : el.linuxCommand.value;
  await api(`/api/process/${name}/start`, {
    method: "POST",
    body: JSON.stringify({ command }),
  });
}

async function stopProcess(name) {
  await api(`/api/process/${name}/stop`, { method: "POST", body: "{}" });
}

function renderStatus(name, status) {
  const target = name === "windows" ? el.windowsStatus : el.linuxStatus;
  target.classList.toggle("running", Boolean(status.running));
  target.classList.toggle("error", !status.running && status.exit_code && status.exit_code !== 0);
  const prefix = name === "windows" ? "Windows" : "Linux";
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
  const item = status.linux.last_important;
  if (!item) {
    el.lastLinuxEvent.textContent = "无事件";
    return;
  }
  el.lastLinuxEvent.textContent = `${item.time} ${item.level}`;
}

async function refreshStatus() {
  const status = await api("/api/status");
  renderStatus("windows", status.windows);
  renderStatus("linux", status.linux);
  renderLastLinuxEvent(status);
}

async function refreshLogs() {
  const [windows, linux, linuxImportant] = await Promise.all([
    api("/api/logs/windows?limit=500"),
    api("/api/logs/linux?limit=500"),
    api("/api/logs/linux?important=1&limit=120"),
  ]);
  renderPlainLog(el.windowsLog, windows);
  renderPlainLog(el.linuxLog, linux);
  renderImportantLog(linuxImportant);
}

function renderPlainLog(target, lines) {
  const text = lines.map((line) => `[${line.time}] ${line.text}`).join("\n");
  if (target.textContent !== text) {
    const shouldStick = target.scrollTop + target.clientHeight >= target.scrollHeight - 20;
    target.textContent = text;
    if (shouldStick) target.scrollTop = target.scrollHeight;
  }
}

function renderImportantLog(lines) {
  if (!lines.length) {
    el.linuxImportantLog.innerHTML = '<div class="empty">等待 Linux 关键事件</div>';
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

el.previewImage.addEventListener("load", () => {
  state.previewReady = true;
  el.previewImage.classList.add("ready");
  el.previewEmpty.style.display = "none";
});

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.activePreview = button.dataset.preview;
    state.previewReady = false;
    state.streamConnected = false;
    el.previewImage.classList.remove("ready");
    el.previewEmpty.style.display = "block";
    el.previewImage.removeAttribute("src");
    refreshPreview().catch(console.error);
  });
});

document.getElementById("saveWindows").addEventListener("click", () => saveCommands().catch(alertError));
document.getElementById("saveLinux").addEventListener("click", () => saveCommands().catch(alertError));
document.getElementById("startWindows").addEventListener("click", () =>
  startProcess("windows").catch(alertError),
);
document.getElementById("startLinux").addEventListener("click", () =>
  startProcess("linux").catch(alertError),
);
document.getElementById("stopWindows").addEventListener("click", () =>
  stopProcess("windows").catch(alertError),
);
document.getElementById("stopLinux").addEventListener("click", () =>
  stopProcess("linux").catch(alertError),
);

function alertError(error) {
  alert(error.message || String(error));
}

async function tick() {
  try {
    await Promise.all([refreshStatus(), refreshLogs(), refreshPreview()]);
  } catch (error) {
    console.error(error);
  }
}

loadCommands()
  .then(tick)
  .catch((error) => {
    console.error(error);
    alertError(error);
  });

setInterval(tick, 1000);
