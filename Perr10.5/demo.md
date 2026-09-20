# Задание 10.5 — как показать работу

## Доступ для проверяющего

| Что | Адрес |
|---|---|
| Редактор n8n | <https://sintaition.sintaris.net> |
| Вебхук цепочки | `POST https://sintaition.sintaris.net/webhook/zerocoder-autopost` |
| Канал с постами | <https://t.me/business_ai_automate> |

Вход в редактор — почта `info@sintaris.net`, пароль лежит на сервере в `~/n8n-new/.env`
(строка `N8N_OWNER_PASSWORD`). В открытый доступ пароль не выкладывается: у владельца
n8n есть права на все ключи, включая ключ OpenAI.

Если проверяющему нужен доступ в редактор — лучше завести ему отдельного пользователя
в n8n (Settings → Users), а не отдавать пароль владельца.

## Показать за одну минуту

Проверяющему не нужен доступ никуда: достаточно одного запроса.

```bash
curl -X POST https://sintaition.sintaris.net/webhook/zerocoder-autopost \
  -H "Content-Type: application/json" \
  -d '{"topic": "Маркировка CE: что проверяет уполномоченный представитель"}'
```

В ответ приходит ссылка на только что опубликованный пост:

```json
{"ok": true, "published": true, "message_id": 4,
 "url": "https://t.me/business_ai_automate/4"}
```

То же самое скриптом из задания:

```bash
cd Perr10.5
N8N_POST_WEBHOOK=https://sintaition.sintaris.net/webhook/zerocoder-autopost \
  python main.py "Маркировка CE: что проверяет уполномоченный представитель"
```

Не отправляя запрос, а только посмотреть, что уйдёт: добавить `--dry-run`.

## Скриншоты для отчёта

Задание просит пять вещей. Четыре из них — снимки экрана, их нужно сделать вручную.
Класть в `Perr10.5/results/screenshots/`.

| # | Что снять | Где это открыть |
|---|---|---|
| 1 | Контейнер n8n со статусом Running | Docker Desktop, либо на сервере `docker ps` — вывод уже сохранён в `results/n8n-состояние.txt` |
| 2 | Схема цепочки целиком | <https://sintaition.sintaris.net> → Workflows → «Zerocoder 10.5 — Автопостинг в Telegram» |
| 3 | Executions со статусом Success | там же → вкладка Executions, запуск №4 |
| 4 | Опубликованный пост | <https://t.me/business_ai_automate/4> |

Пятый пункт — файл `main.py` — уже в папке задания.

## Если что-то не работает

| Признак | Что проверить |
|---|---|
| Адрес не открывается | туннель: на SintAItion `systemctl --user status taris-tunnel-n8n` |
| «webhook is not registered» | цепочка выключена — включить в редакторе или запустить `deploy_workflow.py` |
| Ответ 200, но поста нет | `last_run.py` покажет, на каком узле остановилось |
| «chat not found» | идентификатор канала должен начинаться с `-100` |

Пересоздать контейнер после правки настроек: на сервере `bash ~/n8n-new/run.sh`.
Данные при этом не теряются — они в PostgreSQL.
