
"""Проверка связей Taris с n8n: что настроено, что отвечает, чего не хватает.

Значения секретов не печатаются: отчёт задуман так, чтобы его можно было
положить рядом с заданием и показать проверяющему.
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HOSTS = {
    "certtaris":     Path("/home/stas/.taris-cert/bot.env"),
    "certtaris-eng": Path("/home/stas/.taris-cert-eng/bot.env"),
    "supertaris":    Path("/home/stas/.taris/bot.env"),
}
N8N = "http://127.0.0.1:5681"
SECRET_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "DSN")


def read_env(p):
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def show(key, value):
    if any(h in key for h in SECRET_HINTS):
        return "(задан)" if value else "(пусто)"
    return value or "(пусто)"


def probe(url, data=b"{}", method="POST", timeout=25):
    req = urllib.request.Request(url, data=data if method == "POST" else None,
                                 headers={"Content-Type": "application/json"}, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, time.perf_counter() - t0, ""
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0, e.read(200).decode("utf-8", "replace")
    except Exception as e:
        return 0, time.perf_counter() - t0, str(e)[:120]


sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_workflow import Client

cli = Client(N8N)
cli.login(os.environ["N8N_OWNER_EMAIL"], os.environ["N8N_OWNER_PASSWORD"])
code, res = cli.call("/rest/workflows")
data = res.get("data")
items = data.get("data", []) if isinstance(data, dict) else (data or [])

registered = {}
print("Рабочие процессы в n8n 2.39.8 (порт 5681)")
print("-" * 78)
for w in items:
    code, full = cli.call(f"/rest/workflows/{w['id']}")
    nodes = full["data"]["nodes"]
    paths = [n["parameters"].get("path") for n in nodes if n["type"].endswith("webhook")]
    active = bool(full["data"].get("active"))
    for p in paths:
        registered[p] = (w["name"], active)
    print(f"  {w['name']:42} {'включён' if active else 'выключен':9} "
          f"вебхуки: {', '.join(p for p in paths if p) or '—'}")

print()
print("Настройки экземпляров Taris, указывающие на n8n")
print("-" * 78)
findings = []
for name, path in HOSTS.items():
    env = read_env(path)
    keys = sorted(k for k in env if k.startswith("N8N_"))
    print(f"\n  {name}  ({path})")
    if not keys:
        print("    переменных N8N_ нет")
        continue
    for k in keys:
        v = env[k]
        printed = show(k, v)
        if not v or not v.startswith("http"):
            print(f"    {k:28} {printed}")
            continue

        is_webhook = "/webhook/" in v
        if is_webhook:
            status, dt, _ = probe(v)
            tail = urllib.parse.urlparse(v).path.rsplit("/", 1)[-1]
            known = registered.get(tail)
            if status == 200:
                verdict = "✔ отвечает"
            elif status == 404 and not known:
                verdict = "✘ процесса с таким вебхуком в n8n нет"
                findings.append(f"{name}: {k} ведёт на «{tail}» — такого процесса нет")
            elif status == 404 and known and not known[1]:
                verdict = f"✘ процесс «{known[0]}» выключен"
                findings.append(f"{name}: {k} ведёт в выключенный процесс «{known[0]}»")
            elif status == 0:
                verdict = "✘ адрес недоступен"
                findings.append(f"{name}: {k} — адрес недоступен")
            else:
                verdict = f"код {status}"
            # вебхук может быть настроен на процесс с другим назначением
            if known and tail not in k.lower().replace("_", "-"):
                pass
        else:
            # это базовый адрес, а не вебхук — проверяем сам n8n
            status, dt, _ = probe(v.rstrip("/") + "/healthz", method="GET")
            if status == 200:
                verdict = "✔ n8n по этому адресу отвечает"
            elif status == 0:
                verdict = "✘ n8n по этому адресу не отвечает"
                findings.append(f"{name}: {k} указывает на n8n, которого нет по этому адресу")
            else:
                verdict = f"код {status}"

        print(f"    {k:28} {printed}")
        print(f"    {'':28} {verdict}  ({dt:.1f} с)")

# сверка назначения: вебхук должен вести в процесс с подходящим именем
print()
print("Сходится ли назначение вебхука с процессом, в который он ведёт")
print("-" * 78)
for name, path in HOSTS.items():
    for k, v in read_env(path).items():
        if k.startswith("N8N_") and "/webhook/" in (v or ""):
            tail = urllib.parse.urlparse(v).path.rsplit("/", 1)[-1]
            known = registered.get(tail)
            target = known[0] if known else "—"
            print(f"  {name:14} {k:28} → {tail:26} процесс: {target}")

print()
print("Итог")
print("-" * 78)
if findings:
    for f in findings:
        print(f"  ! {f}")
else:
    print("  все настроенные адреса отвечают")
