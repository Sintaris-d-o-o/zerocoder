# N3 — Реализация ИИ-ассистента для обработки вакансий

## Что реализовано

Полноценный n8n workflow `n3-hr-assistant.workflow.json` из 18 нод, который:
1. Получает письма с PDF-резюме через Gmail-триггер
2. Извлекает текст из PDF
3. Структурирует данные кандидата через GPT-4o
4. Считывает открытые вакансии из Google Sheets
5. Запускает ИИ-мэтчинг (0–100 баллов) через GPT-4o
6. Параллельно сохраняет профиль кандидата в базу (второй лист)
7. Отправляет персонализированное письмо-приглашение или письмо-отказ
8. Помечает исходное письмо в Gmail звёздочкой

---

## Структура файлов

```
workflows/
  n3-hr-assistant.workflow.json   ← workflow для импорта в n8n
scripts/
  publish-hr-workflow.ps1         ← скрипт деплоя
.env                              ← добавлены N8N_HR_WORKFLOW_ID и N8N_HR_SHEETS_DOC_ID
```

---

## Схема workflow (18 нод)

```
[Gmail Trigger]
      │  скачивает PDF-вложение
      ▼
[Extract PDF Text]
      │  $json.text — сырой текст резюме
      ▼
[Build Extraction Body]  ← Code: собирает JSON-тело для OpenAI
      ▼
[Extract Candidate Info]  ← HTTP → OpenAI /chat/completions
      │  response_format: json_object, temp 0.1
      ▼
[Parse Candidate JSON]  ← Code: парсит ответ, добавляет emailFrom/messageId
      ▼
[Get Vacancies]  ← Google Sheets, лист «Вакансии»
      │  возвращает N строк
      ▼
[Aggregate Vacancies]  ← Code (allItems): собирает массив [{row_number, title, description}]
      ▼
[Build Matching Body]  ← Code: собирает системный промпт + данные в JSON-тело
      ▼
[Run Matching]  ← HTTP → OpenAI /chat/completions
      │  response_format: json_object, temp 0.3
      ▼
[Parse Matching Result]  ← Code: извлекает matches (score≥60), hasMatch
      │
      ├──→ [Save Candidate]     ← Google Sheets Append, лист «Кандидаты»
      │
      └──→ [Has Match?]  ← If: $json.hasMatch === true
                ├── TRUE  → [Wait Invite 45 мин] → [Send Invite] → [Label ★]
                └── FALSE → [Wait Reject 45 мин] → [Send Reject] → [Label ★]
```

---

## Шаг 1 — Подготовка Google Sheets

Создайте Google Spreadsheet с двумя листами:

### Лист «Вакансии» (обязательные заголовки в строке 1)
| Должность | Описание | Требования | Зарплата |
|---|---|---|---|
| Frontend Developer | React, Redux, TypeScript, remote | 3+ лет опыта, middle/senior | 150 000 – 200 000 |

### Лист «Кандидаты» (обязательные заголовки)
| Имя | Позиция | Навыки | Опыт | Уровень | Город | Страна | Зарплата | Email | Телефон | Результат | Дата |
|---|---|---|---|---|---|---|---|---|---|---|---|

Скопируйте ID документа из URL:
```
https://docs.google.com/spreadsheets/d/  ██████████████████████  /edit
                                          ↑ это и есть DOC_ID
```

Запишите его в `.env`:
```
N8N_HR_SHEETS_DOC_ID=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms
```

---

## Шаг 2 — Настройка credentials в n8n UI

Перейдите на `https://automata.dev2null.de` → **Settings → Credentials**.

### Gmail OAuth2
1. Создайте credential типа **Gmail OAuth2**
2. Настройте через Google Cloud Console:
   - Включите **Gmail API**
   - Создайте OAuth2 credentials (тип: Web Application)
   - Redirect URI: `https://automata.dev2null.de/rest/oauth2-credential/callback`
3. Авторизуйтесь
4. Скопируйте ID credential (из адресной строки при редактировании)

### Google Sheets OAuth2
1. Создайте credential типа **Google Sheets OAuth2**
2. Используйте тот же Google Cloud project
3. Включите **Google Sheets API**
4. Авторизуйтесь
5. Скопируйте ID credential

---

## Шаг 3 — Замена плейсхолдеров в workflow JSON

Откройте [workflows/n3-hr-assistant.workflow.json](../workflows/n3-hr-assistant.workflow.json) и замените:

| Плейсхолдер | Что подставить |
|---|---|
| `REPLACE_WITH_GMAIL_CREDENTIAL_ID` | ID Gmail OAuth2 credential из n8n (встречается 4 раза) |
| `REPLACE_WITH_GSHEETS_CREDENTIAL_ID` | ID Google Sheets OAuth2 credential (встречается 2 раза) |
| `REPLACE_WITH_GOOGLE_SHEETS_DOC_ID` | ID документа Google Sheets (встречается 2 раза) |

Быстрая замена в PowerShell:
```powershell
$wf = Get-Content -Raw "workflows/n3-hr-assistant.workflow.json"
$wf = $wf -replace 'REPLACE_WITH_GMAIL_CREDENTIAL_ID',   'ВАШ_GMAIL_CREDENTIAL_ID'
$wf = $wf -replace 'REPLACE_WITH_GSHEETS_CREDENTIAL_ID', 'ВАШ_SHEETS_CREDENTIAL_ID'
$wf = $wf -replace 'REPLACE_WITH_GOOGLE_SHEETS_DOC_ID',  'ВАШ_SHEETS_DOC_ID'
Set-Content "workflows/n3-hr-assistant.workflow.json" $wf
```

---

## Шаг 4 — Деплой workflow в n8n

### Создание нового workflow (первый раз)
```powershell
cd d:\Projects\workspace\learning\zerocoder
.\scripts\publish-hr-workflow.ps1 -Create
```
Скрипт создаст workflow через API, сохранит ID в `.env` → `N8N_HR_WORKFLOW_ID`.

### Обновление существующего workflow
```powershell
.\scripts\publish-hr-workflow.ps1
```

### Деплой + активация
```powershell
.\scripts\publish-hr-workflow.ps1 -Activate
```

---

## Шаг 5 — Проверка в n8n UI

1. Откройте workflow в браузере: `https://automata.dev2null.de/workflow/N8N_HR_WORKFLOW_ID`
2. Убедитесь, что все ноды показывают зелёный статус credentials
3. Запустите **Gmail Trigger → Execute Node** и отправьте тестовое письмо с PDF

**Тестовое письмо:**
- **Тема:** `Frontend разработчик`
- **Текст:** `Здравствуйте! Высылаю своё резюме.`
- **Вложение:** PDF с резюме

---

## Шаг 6 — Активация

После успешного теста активируйте через UI (кнопка Active в правом верхнем углу) или через скрипт:
```powershell
.\scripts\publish-hr-workflow.ps1 -Activate
```

---

## Описание ключевых нод

### Gmail Trigger
- Polling: каждую минуту
- Фильтр: **только непрочитанные**
- `simple: false` — полная структура с `from.value[0].address`
- `downloadAttachments: true` — PDF в `$binary.attachment_0`

### Extract PDF Text
- Нода `extractFromFile`, operation `pdf`
- Читает `attachment_0`, возвращает `$json.text`

### Build Extraction Body + Extract Candidate Info
- Code-нода формирует валидный JSON для OpenAI API
- Модель `gpt-4o`, temperature `0.1`, `response_format: json_object`
- 13 полей: full_name, sex, birth_date, contact_phone, email, country, city, citizenship, position, employment, experience, skills, salary

### Aggregate Vacancies
- Code-нода (`runOnceForAllItems`) — собирает N строк GSheets в один массив
- Обращается к предыдущим нодам через `$('Parse Candidate JSON').first().json`

### Build Matching Body + Run Matching
- Системный промпт ~2 КБ: нормализация, семейства должностей, уровни сениорности
- Скоринг по 7 критериям (0–100), порог 60
- Температура `0.3` для стабильного JSON

### Parse Matching Result
- Фильтрует `matches` по `score >= 60`
- Устанавливает `hasMatch: boolean`
- **Параллельный выход** на `Save Candidate` и `Has Match?`

### Has Match? (If Node)
- `TRUE` вывод `[0]` → ветка приглашения
- `FALSE` вывод `[1]` → ветка отказа

### Wait Nodes (45 минут)
- Имитирует «человеческое» время обработки
- Защита от детектирования автоматизации кандидатами

---

## Поля сохраняемые в Google Sheets (лист «Кандидаты»)

| Колонка | Источник |
|---|---|
| Имя | `candidateSummary.full_name` |
| Позиция | `candidateSummary.position_raw` |
| Навыки | `candidateSummary.normalized_skills.join(', ')` |
| Опыт | `candidateSummary.experience_years` |
| Уровень | `candidateSummary.seniority` |
| Город | `candidateSummary.city` |
| Страна | `candidateSummary.country` |
| Зарплата | `candidateSummary.salary_expectation_rub_per_month` |
| Email | `emailFrom` (из Gmail Trigger) |
| Телефон | `candidate.contact_phone` |
| Результат | `Приглашение` / `Отказ` |
| Дата | `new Date().toISOString()` |

---

## TODO / Следующие шаги

- [ ] **Защита от дублирования** — добавить ноду Google Sheets Read перед мэтчингом, проверять по полю Email
- [ ] **Случайная задержка** — заменить фиксированные 45 мин на `={{ 30 + Math.floor(Math.random() * 60) }}` минут
- [ ] **Кастомный Gmail label** — создать метку `HR Auto Response` в Gmail и заменить `STARRED`
- [ ] **Telegram-уведомление** — алерт HR-менеджеру при `score > 85`
- [ ] **Error Handler** — добавить нodu Error Trigger с уведомлением при сбое
