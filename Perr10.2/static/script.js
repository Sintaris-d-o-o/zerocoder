const $ = (id) => document.getElementById(id);
const form = $("form"), go = $("go");
const box = { status: $("status"), result: $("result"), error: $("error") };

let lastSeed = null;
let pollTimer = null;

function show(which) {
  Object.entries(box).forEach(([k, el]) => el.classList.toggle("hidden", k !== which && which !== "none"));
}

function fail(message) {
  clearInterval(pollTimer);
  box.error.textContent = message;
  show("error");
  go.disabled = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const company = $("company").value.trim();
  if (!company) { fail("Введите название компании"); return; }

  go.disabled = true;
  show("status");
  $("status-text").textContent = "Отправляем запрос…";
  $("status-prompt").textContent = "";

  let started;
  try {
    const resp = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        company,
        extra: $("extra").value,
        style: $("style").value,
        format: $("format").value,
        seed: $("seed").value,
      }),
    });
    started = await resp.json();
    if (!resp.ok) { fail(started.error || "Не удалось запустить генерацию"); return; }
  } catch (err) {
    fail("Сервер недоступен: " + err.message);
    return;
  }

  $("status-prompt").textContent = started.prompt;
  const t0 = Date.now();

  pollTimer = setInterval(async () => {
    let job;
    try {
      const r = await fetch(`/status/${started.job_id}`);
      job = await r.json();
      if (!r.ok) { fail(job.error || "Задача потерялась"); return; }
    } catch (err) {
      fail("Потеряна связь с сервером: " + err.message);
      return;
    }

    const secs = Math.round((Date.now() - t0) / 1000);
    if (job.status === "queued" || job.status === "running") {
      const checks = job.polls ? `, проверок статуса: ${job.polls}` : "";
      $("status-text").textContent = `Генерируем… ${secs} с${checks}`;
      return;
    }

    clearInterval(pollTimer);
    go.disabled = false;

    if (job.status === "error") { fail(job.message || "Генерация не удалась"); return; }

    lastSeed = job.seed;
    $("result-img").src = `/image/${job.filename}?t=${Date.now()}`;
    $("result-download").href = `/download/${job.filename}`;
    $("result-info").textContent =
      `Готово за ${job.elapsed_s} с · зерно ${job.seed} · файл ${job.filename}`;
    show("result");
  }, 1000);
});

$("result-repeat").addEventListener("click", () => {
  if (lastSeed != null) $("seed").value = lastSeed;
  form.dispatchEvent(new Event("submit"));
});
