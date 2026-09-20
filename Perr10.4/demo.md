# Демонстрация задания 10.4

## Что показывать проверяющему

1. [результаты.md](результаты.md) — отчёт, сверка по трём частям задания, ответы на три
   вопроса о архитектуре.
2. [02-как-это-реализовано-в-certtaris.md](02-как-это-реализовано-в-certtaris.md) — разбор с
   цитатами кода, путями и номерами строк.
3. [results/screenshots/](results/screenshots/) — 17 снимков работающей системы.
4. [results/services.txt](results/services.txt) — живой вывод состояния служб с сервера.

## Ключевые доказательства

| Требование задания | Чем доказано |
|---|---|
| Бот обновляет одно сообщение | [Request.png](results/screenshots/Request.png) и [Request1.png](results/screenshots/Request1.png) — та же метка времени, заглушка заменена ответом |
| Прогресс долгой задачи на сайте | [web_documents_reindex_progress.png](results/screenshots/web_documents_reindex_progress.png) — «2/49» и имя текущего документа |
| Скачивание файла | [web_notes_download.png](results/screenshots/web_notes_download.png) и следующие два |
| Процессы работают 24/7 | [results/services.txt](results/services.txt) — четверо суток непрерывно, автозапуск при загрузке |
| Стадии обработки со временем | там же, журнал бота: распознавание, модель, синтез речи |

## Живая демонстрация

- **Сайт:** `https://agents.sintaris.net/certtaris/` — нужен вход.
- **Бот:** в Telegram, доступ по списку разрешённых пользователей.

Проверяющему без доступа достаточно скриншотов и вывода служб: они сняты с работающей системы.

## Как получен вывод служб

Подключением к серверу и запуском команд состояния:

```bash
systemctl --user status taris-cert-web taris-cert-telegram --no-pager
systemctl --user list-units 'taris*' --no-pager
journalctl --user -u taris-cert-telegram -n 6 --no-pager
```

Файл [results/services.txt](results/services.txt) — это их вывод целиком, без правок.
