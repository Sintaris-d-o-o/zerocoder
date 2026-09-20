# Задание 10.6 — как показать работу

## Доступ

| Что | Адрес |
|---|---|
| Редактор n8n | <https://sintaition.sintaris.net> |
| Вебхук цепочки | `POST https://sintaition.sintaris.net/webhook/zerocoder-autopost-image` |
| Канал с постами | <https://t.me/business_ai_automate> |

Токен для заголовка `Authorization` лежит на сервере в `~/n8n-new/.env`
(`N8N_WEBHOOK_TOKEN`). Вход в редактор — `info@sintaris.net`, пароль там же
(`N8N_OWNER_PASSWORD`).

## Показать за две минуты

Сначала показать, что без токена не пускает:

```bash
curl -i -X POST https://sintaition.sintaris.net/webhook/zerocoder-autopost-image \
  -H "Content-Type: application/json" \
  -d '{"topic": "Технический файл медизделия"}'
```

Ответ — `403 Forbidden`, «Authorization data is wrong!», и приходит он мгновенно:
цепочка не запускается, деньги не тратятся.

Теперь с токеном:

```bash
curl -X POST https://sintaition.sintaris.net/webhook/zerocoder-autopost-image \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $N8N_WEBHOOK_TOKEN" \
  -d '{"topic": "Технический файл медизделия"}'
```

Через 15–20 секунд возвращается ссылка на пост:

```json
{"ok": true, "published": true, "with_image": true,
 "caption_used": true, "message_id": 7,
 "url": "https://t.me/business_ai_automate/7"}
```

То же самое скриптом из задания:

```bash
cd Perr10.6
N8N_POST_WEBHOOK=https://sintaition.sintaris.net/webhook/zerocoder-autopost-image \
  python main.py "Технический файл медизделия"

# и проверка защиты
python main.py --no-token "Технический файл медизделия"
```

Скрипт сам понимает, что при `--no-token` правильный исход — это 403, и возвращает
нулевой код возврата именно в этом случае.

## Скриншоты для отчёта

Задание просит шесть вещей. Пять из них — снимки экрана.

| # | Что снять | Где это открыть |
|---|---|---|
| 1 | Полная схема цепочки | <https://sintaition.sintaris.net> → Workflows → «Zerocoder 10.6 — Автопостинг с картинкой» |
| 2 | Успешное выполнение | там же → Executions → запуск со статусом Success |
| 3 | Пост в канале: текст и картинка | <https://t.me/business_ai_automate/7> |
| 4 | Ответ со статусом 200 | вывод `main.py` или `curl -i` с токеном |
| 5 | Ошибка 403 без токена | вывод `main.py --no-token` или `curl -i` без заголовка |

Шестой пункт — файл `workflow.json` — уже в папке задания.

Текстовый протокол пунктов 4 и 5 сохранён в `results/проверка-защиты.txt`; скриншоты
терминала можно снять с этого вывода.

## Если что-то не работает

| Признак | Что проверить |
|---|---|
| 403 с правильным токеном | значение в `~/n8n-new/.env` и в учётных данных n8n «Zerocoder 10.6 — Bearer» должны совпадать целиком, вместе со словом Bearer |
| 200, но поста нет | `last_run.py` покажет, на каком узле остановилось |
| «Bad request» на картинке | модель серии gpt-image не принимает `response_format` — запрос должен идти HTTP-узлом, а не узлом OpenAI |
| Пост без картинки | Telegram принял `sendPhoto`, но без файла: проверить поле `photo` типа formBinaryData и узел «Image to Binary» |
| Подпись обрезана | текст длиннее 1024 знаков уходит отдельным сообщением — это штатное поведение |

Перезалить цепочку после правки файла: на сервере
`cd ~/n8n-new/zerocoder10.6 && python3 deploy_workflow.py`.
