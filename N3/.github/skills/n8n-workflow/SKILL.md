---
name: n8n-workflow
description: 'Build, edit, debug, or publish n8n workflow JSON files for the automata.dev2null.de instance. Use when: creating a new n8n workflow, adding nodes, fixing broken nodes, replacing HTTP OpenAI calls with native nodes, deploying a workflow via API, configuring credentials, writing Code nodes in JavaScript, setting up triggers (Gmail, Google Sheets, Schedule), implementing AI agents in n8n, publishing and updating workflows, checking n8n status, viewing n8n logs, SSHing to VPS, diagnosing Gmail rate limits, monitoring workflow executions, deactivating or activating workflows, manually executing a workflow, listing executions, reviewing live workflow structure, comparing local vs live workflow, listing all workflows, testing workflows locally or on n8n host, inspecting execution errors and node output. Contains verified node compatibility table, VPS operations reference, and execution management patterns for this specific n8n instance.'
---

# n8n Workflow Skill

## Instance Details

All connection details are in `.env` — read them from there, never hardcode in docs or workflow JSON.

| Setting | `.env` variable |
|---|---|
| Host URL | `N8N_HOST` |
| API base URL | `N8N_API_BASE` |
| API key | `N8N_API_KEY` |
| Execution order | `v1` (hardcoded in workflow settings) |
| Workflow files | `workflows/*.workflow.json` |

---

## ✅ Verified Compatible Nodes

This is the authoritative compatibility table for this n8n instance. Always use these exact `type` and `typeVersion` values.

### Core / Logic

| Node | type | typeVersion | Notes |
|---|---|---|---|
| Schedule Trigger | `n8n-nodes-base.scheduleTrigger` | `1` | |
| Code | `n8n-nodes-base.code` | `2` | JS only; modes: `runOnceForEachItem`, `runOnceForAllItems` |
| If | `n8n-nodes-base.if` | `1` | |
| Switch | `n8n-nodes-base.switch` | `3` | |
| Set | `n8n-nodes-base.set` | `3.4` | |
| Wait | `n8n-nodes-base.wait` | `1.1` | `resume: timeInterval` |
| HTTP Request | `n8n-nodes-base.httpRequest` | `4.2` | |
| Extract From File | `n8n-nodes-base.extractFromFile` | `1` | operation: `pdf` for PDF |

### Google

Credential IDs are stored in `.env` — see `N8N_GSHEETS_CREDENTIAL_ID`, `N8N_GSHEETS_TRIGGER_CREDENTIAL_ID`, `N8N_GMAIL_CREDENTIAL_ID`.

| Node | type | typeVersion | Credential type | `.env` var |
|---|---|---|---|---|
| Google Sheets (read/write/append) | `n8n-nodes-base.googleSheets` | `4.5` or `4.7` | `googleSheetsOAuth2Api` | `N8N_GSHEETS_CREDENTIAL_ID` |
| Google Sheets Trigger | `n8n-nodes-base.googleSheetsTrigger` | `1` | `googleSheetsTriggerOAuth2Api` | `N8N_GSHEETS_TRIGGER_CREDENTIAL_ID` |
| Gmail (send/addLabels) | `n8n-nodes-base.gmail` | `2.1` | `gmailOAuth2` | `N8N_GMAIL_CREDENTIAL_ID` |
| Gmail Trigger | `n8n-nodes-base.gmailTrigger` | `1` | `gmailOAuth2` | `N8N_GMAIL_CREDENTIAL_ID` |

### AI / LLM

Credential ID is in `.env` — see `N8N_OPENAI_CREDENTIAL_ID`.

| Node | type | typeVersion | Credential type | `.env` var |
|---|---|---|---|---|
| OpenAI — Message a model | `@n8n/n8n-nodes-langchain.openAi` | `2.1` | `openAiApi` | `N8N_OPENAI_CREDENTIAL_ID` |

---

## ❌ Known Broken / Incompatible Nodes

| Broken type | Use instead |
|---|---|
| `n8n-nodes-base.openAi` | `@n8n/n8n-nodes-langchain.openAi` v2.1 |
| `@n8n/n8n-nodes-langchain.openAi` v1.x | Use v2.1 |

---

## OpenAI Node — Correct Structure

**Critical**: The `@n8n/n8n-nodes-langchain.openAi` v2.1 node uses `responses.values`, NOT `messages.values`.

```json
{
  "type": "@n8n/n8n-nodes-langchain.openAi",
  "typeVersion": 2.1,
  "credentials": {
    "openAiApi": {
      "id": "<N8N_OPENAI_CREDENTIAL_ID from .env>",
      "name": "<N8N_OPENAI_CREDENTIAL_NAME from .env>"
    }
  },
  "parameters": {
    "modelId": {
      "__rl": true,
      "value": "gpt-4o",
      "mode": "list",
      "cachedResultName": "gpt-4o"
    },
    "responses": {
      "values": [
        { "role": "system", "content": "Your system prompt here" },
        { "content": "=User message with {{$json.myField}} expression" }
      ]
    },
    "options": {
      "temperature": 0.3
    }
  }
}
```

**Output**: The response text is at `$json.content` (NOT `$json.choices[0].message.content`).  
To get JSON output: instruct in prompt to return ONLY JSON, then `JSON.parse($json.content)` in a Code node.

---

## Google Sheets Node — Correct Structure

```json
{
  "type": "n8n-nodes-base.googleSheets",
  "typeVersion": 4.5,
  "credentials": {
    "googleSheetsOAuth2Api": {
      "id": "<N8N_GSHEETS_CREDENTIAL_ID from .env>",
      "name": "<N8N_GSHEETS_CREDENTIAL_NAME from .env>"
    }
  },
  "parameters": {
    "operation": "read",
    "documentId": {
      "__rl": true,
      "value": "YOUR_SHEET_ID",
      "mode": "id"
    },
    "sheetName": {
      "__rl": true,
      "value": "SheetName",
      "mode": "name"
    },
    "options": {}
  }
}
```

For `append` operations, use `"operation": "append"` and add a `"columns"` mapping block.

---

## Code Node Patterns

### Read from previous non-adjacent node (allItems mode)
```javascript
const prev = $('Node Name').first().json;
// or all items:
const allItems = $('Node Name').all().map(i => i.json);
```

### allItems mode — iterate input
```javascript
// mode: runOnceForAllItems
const results = items.map(item => ({ json: { ...item.json, processed: true } }));
return results;
```

### Parse OpenAI response
```javascript
const content = $json.content || '';
let parsed;
try {
  parsed = JSON.parse(content);
} catch(e) {
  throw new Error('JSON parse failed: ' + content.slice(0, 200));
}
return [{ json: parsed }];
```

---

## Workflow JSON Structure

```json
{
  "name": "Workflow Name",
  "settings": { "executionOrder": "v1" },
  "nodes": [...],
  "connections": {
    "NodeA": {
      "main": [[{ "node": "NodeB", "type": "main", "index": 0 }]]
    }
  }
}
```

**Strip before POST/PUT**: `id`, `active`, `createdAt`, `updatedAt`, `versionId`  
**POST** (create): `/api/v1/workflows`  
**PUT** (update): `/api/v1/workflows/{id}`  
**Activate**: `POST /api/v1/workflows/{id}/activate`

---

## Deploy Procedure

### Any workflow — deploy / create / activate
```powershell
# Update an existing workflow
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json

# Create new in n8n (auto-saves ID to .env as N8N_WF_MY_WORKFLOW_ID)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Create

# Update + activate
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Activate

# Check status + recent executions
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Status

# Deactivate
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Deactivate

# Delete (requires confirmation)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Delete
```

**EnvKey convention**: `publish.ps1` auto-derives the `.env` key from the filename:
- `my-workflow.workflow.json` → `N8N_WF_MY_WORKFLOW_ID`
- Override with `-EnvKey N8N_HR_WORKFLOW_ID` for legacy workflows

Workflow-specific wrappers (backward compat):
- `scripts/publish-hr-workflow.ps1` — HR assistant (`N8N_HR_WORKFLOW_ID`)
- `scripts/publish-workflow.ps1` — Content factory (`N8N_WORKFLOW_ID`)

### Manual API deploy (Python)
```python
import json, urllib.request

env = {}
with open('.env', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip()

API_KEY  = env['N8N_API_KEY']
API_BASE = env['N8N_API_BASE']   # https://automata.dev2null.de/api
WF_ID    = env['N8N_WF_MY_WORKFLOW_ID']  # use actual key for your workflow

with open('workflows/my.workflow.json') as f:
    wf = json.load(f)

for field in ['id', 'active', 'createdAt', 'updatedAt', 'versionId']:
    wf.pop(field, None)

req = urllib.request.Request(
    f'{API_BASE}/v1/workflows/{WF_ID}',
    data=json.dumps(wf).encode(),
    headers={'X-N8N-API-KEY': API_KEY, 'Content-Type': 'application/json'},
    method='PUT'
)
with urllib.request.urlopen(req) as resp:
    result = json.load(resp)
    print('Updated:', result['id'])
```

---

## Connections Format

For branching (If node):
- `"main": [[...true branch...], [...false branch...]]` — index 0 = true, index 1 = false

For parallel outputs (one node → two nodes):
```json
"NodeA": {
  "main": [[
    { "node": "NodeB", "type": "main", "index": 0 },
    { "node": "NodeC", "type": "main", "index": 0 }
  ]]
}
```

---

## Quality Review Checklist (Before Publishing)

Run through every item before calling the publish scripts:

1. **All nodes connected** — every node except terminal leaf nodes has at least one outgoing connection; every node except the trigger has an incoming connection
2. **Connection keys match node names** — connection object keys must exactly match the `"name"` field of the target node (case-sensitive)
3. **No sequential Code nodes** — two Code nodes directly connected with no other node in between should be merged into one
4. **No unicode escapes in strings** — write system prompts and text directly in UTF-8; if you see `\u0422\u044b` instead of `Ты`, the file was saved with `ensure_ascii=True`. Fix: use `json.dump(..., ensure_ascii=False)`
5. **`$('Node Name')` references** — every `$('...')` call in jsCode must reference the current actual node name; update after any rename
6. **System prompt in OpenAI node** — put system prompts directly in the OpenAI node `responses.values[0]` content, not in an upstream Code node variable
7. **Credential IDs from `.env`** — all `id` fields in `credentials` blocks come from `N8N_*_CREDENTIAL_ID` vars in `.env`
8. **Strip meta fields before deploy** — remove `id`, `active`, `createdAt`, `updatedAt`, `versionId` before POST or PUT
9. **Check n8n UI after deploy** — load the workflow in the n8n canvas, confirm all nodes render with icons (not empty gray boxes), and all edges are visible

---

## Common Mistakes to Avoid

1. **Wrong OpenAI node** — always `@n8n/n8n-nodes-langchain.openAi` v2.1, never `n8n-nodes-base.openAi`
2. **Wrong OpenAI params key** — `responses.values`, not `messages.values`
3. **Wrong OpenAI output path** — `$json.content`, not `$json.choices[0].message.content`
4. **Sending `active` on POST** — n8n rejects it as read-only
5. **Using `authentication: oAuth2`** in Google Sheets — just set the `credentials` block; no `authentication` parameter needed in v4.5+
6. **Referencing renamed nodes** — if you rename a Code node, update all `$('Old Name')` references in downstream Code nodes

---

## VPS & n8n Operations

### SSH Access

All SSH details are in `.env`:

| Setting | `.env` var | Actual value |
|---|---|---|
| SSH host | `SSH_HOST` | `dev2null.de` |
| SSH user | `SSH_USER` | `stas` |
| SSH port | `SSH_PORT` | `22` |
| SSH key | `SSH_KEY_PATH` | `~/.ssh/id_ed25519` |

### Docker Container

**Container name: `n8n-docker-n8n-1`** — NOT `n8n_docker` (old name, will fail with "No such container").

```bash
# Check container status + server time
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker ps | grep n8n && date"

# View recent logs (last 2 hours, last 80 lines)
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 --since 2h 2>&1 | tail -80"

# View absolute latest lines
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 2>&1 | tail -30"

# Search logs for a specific workflow or error keyword
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 2>&1 | grep '<WORKFLOW_ID>' | tail -20"
```

### Check Workflow Status via API

Read `.env` values first; never hardcode credentials in scripts.

```python
import urllib.request, json

API_BASE = 'https://automata.dev2null.de/api/v1'
API_KEY  = '<N8N_API_KEY from .env>'
WF_ID    = '<N8N_WF_YOUR_WORKFLOW_ID from .env>'
headers  = {'X-N8N-API-KEY': API_KEY}

# Workflow active status
req = urllib.request.Request(f'{API_BASE}/workflows/{WF_ID}', headers=headers)
with urllib.request.urlopen(req) as r:
    wf = json.loads(r.read())
print(f'active={wf["active"]}  name={wf["name"]}')

# Recent executions (any status)
req2 = urllib.request.Request(f'{API_BASE}/executions?workflowId={WF_ID}&limit=5', headers=headers)
with urllib.request.urlopen(req2) as r:
    execs = json.loads(r.read())
for e in execs['data']:
    print(f'  id={e["id"]} status={e["status"]} mode={e["mode"]} startedAt={e["startedAt"]}')

# Only failed executions
req3 = urllib.request.Request(f'{API_BASE}/executions?workflowId={WF_ID}&status=error&limit=10', headers=headers)
with urllib.request.urlopen(req3) as r:
    errs = json.loads(r.read())
print(f'Failed: {len(errs["data"])}')
```

### Deactivate / Activate Workflow via API

```python
import urllib.request, json

API_BASE = 'https://automata.dev2null.de/api/v1'
API_KEY  = '<N8N_API_KEY from .env>'
WF_ID    = '<N8N_WF_YOUR_WORKFLOW_ID from .env>'
headers  = {'X-N8N-API-KEY': API_KEY, 'Content-Type': 'application/json'}

# Deactivate
req = urllib.request.Request(f'{API_BASE}/workflows/{WF_ID}/deactivate',
      data=b'{}', headers=headers, method='POST')
with urllib.request.urlopen(req) as r:
    print('Deactivated:', json.loads(r.read()).get('active'))

# Activate
req = urllib.request.Request(f'{API_BASE}/workflows/{WF_ID}/activate',
      data=b'{}', headers=headers, method='POST')
with urllib.request.urlopen(req) as r:
    print('Activated:', json.loads(r.read()).get('active'))
```

### Known Issue: Gmail Rate Limit on Container Restart

**Symptom**: `User-rate limit exceeded. Retry after 2026-04-06T07:xx:xxZ` repeating in logs every 5 minutes.

**Cause**: n8n performs a rapid-fire polling burst on startup (~every 1 min, 5–6 cycles) regardless of configured interval. Each restart burns through the per-user Gmail API quota.

**Diagnosis**:
1. Check logs for `User-rate limit exceeded` — the `Retry after` timestamp tells you when Google's window resets
2. Check container uptime: `sudo docker ps | grep n8n` — short uptime (e.g. "Up 17 minutes") = recent restart = quota burn
3. Check executions API: if last execution is from April 4 but workflow shows `active: True`, Gmail quotas are blocking it

**Recovery**:
- Google's rate limit uses a **sliding window** — each failed poll attempt resets the timer. n8n does NOT honor `Retry-After` and keeps polling on its own schedule, which means the window can never expire while the workflow is active.
- **The only effective fix: deactivate the workflow immediately** to stop all polling:
  ```bash
  python3 -c "
  import urllib.request, json
  env = {}
  [env.update({k.strip(): v.strip()}) for line in open('.env', encoding='utf-8') for k, v in [line.strip().split('=',1)] if '=' in line and not line.startswith('#')]
  h = {'X-N8N-API-KEY': env['N8N_API_KEY'], 'Content-Type': 'application/json'}
  WF_ID = env.get('N8N_HR_WORKFLOW_ID') or env['N8N_WF_N3_HR_ASSISTANT_ID']
  urllib.request.urlopen(urllib.request.Request(env['N8N_API_BASE'].rstrip('/')+'/v1/workflows/'+WF_ID+'/deactivate', data=b'{}', headers=h, method='POST'))
  print('Deactivated.')
  "
  ```
- Wait **15–30 minutes** with workflow deactivated (Docker logs will show `Deregistered all crons for workflow`)
- Then reactivate with the publish script: `.\scripts\publish.ps1 -WorkflowFile workflows/n3-hr-assistant.workflow.json -Activate`
- Do NOT restart the container to "fix" it — that triggers another polling burst and resets the window again

**Prevention**: Use `"mode": "everyX", "value": 5, "unit": "minutes"` — NOT `"mode": "everyMinute"` — in Gmail Trigger `pollTimes`. The `everyX` format (not `everyFiveMinutes`) is what actually gets saved by the n8n API:
```json
"pollTimes": {
  "item": [{ "mode": "everyX", "value": 5, "unit": "minutes" }]
}
```

---

## Execution Management

### List Executions via publish.ps1

```powershell
# Last 10 executions
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions

# Last 20 executions
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions -ExecutionsLimit 20
```

Output shows: `id`, `status` (success/error/waiting), `mode` (trigger/manual/webhook), `duration`, `startedAt`. Automatically fetches and displays the last error detail if any failed execution is found.

### List Executions via API (Python)

```python
import urllib.request, json

env = {}
with open('.env', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1); env[k.strip()] = v.strip()

API_BASE = env['N8N_API_BASE'].rstrip('/') + '/v1'
headers  = {'X-N8N-API-KEY': env['N8N_API_KEY']}
WF_ID    = env['N8N_WF_YOUR_WORKFLOW_ID']

# List last 10 executions (all statuses)
req = urllib.request.Request(f'{API_BASE}/executions?workflowId={WF_ID}&limit=10', headers=headers)
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())
for e in data['data']:
    print(f'id={e["id"]} status={e["status"]} mode={e["mode"]} startedAt={e["startedAt"]}')

# Only errors
req = urllib.request.Request(f'{API_BASE}/executions?workflowId={WF_ID}&status=error&limit=5', headers=headers)
with urllib.request.urlopen(req) as r:
    errs = json.loads(r.read())
print(f'Failed executions: {len(errs["data"])}')
```

### Get Execution Details (Node Output)

```python
EXEC_ID = '123'  # from executions list above
req = urllib.request.Request(f'{API_BASE}/executions/{EXEC_ID}', headers=headers)
with urllib.request.urlopen(req) as r:
    detail = json.loads(r.read())

# Error info
err = detail['data']['resultData'].get('error', {})
if err:
    print('Error:', err.get('message'))
    print('Node :', err.get('node', {}).get('name'))

# Node output data
run_data = detail['data']['resultData'].get('runData', {})
for node_name, node_runs in run_data.items():
    for run in node_runs:
        output = run.get('data', {}).get('main', [[]])[0]
        print(f'{node_name}: {len(output)} items')
```

### Manually Execute Workflow (via publish.ps1)

```powershell
# Trigger a manual test run on the VPS via SSH + docker exec
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Execute
```

This runs `docker exec n8n-docker-n8n-1 n8n execute --id=<ID>` on the VPS via SSH. Useful for testing non-trigger workflows or forcing a test run. Not the same as a trigger-based run — the workflow starts from the first node, bypassing the trigger.

### Manually Execute via SSH + docker exec (direct)

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de \
  "sudo docker exec n8n-docker-n8n-1 n8n execute --id=<WORKFLOW_ID> 2>&1 | tail -50"
```

---

## Workflow Review & Inspection

### Review Live Workflow on n8n

```powershell
# Fetch live workflow from n8n, show node inventory, compare with local file
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Review
```

Output:
- Node inventory (type, version, name, credentials used)
- Local vs. live node count & name comparison
- Warnings if local file is out of sync with n8n

### List All Workflows

```powershell
# All workflows on the n8n instance with active status
.\scripts\publish.ps1 -WorkflowFile workflows/any.workflow.json -ListAll
```

Or via API:
```python
req = urllib.request.Request(f'{API_BASE}/workflows?limit=100', headers=headers)
with urllib.request.urlopen(req) as r:
    wfs = json.loads(r.read())
for w in wfs['data']:
    print(f'{"[ACTIVE]" if w["active"] else "[off]   "}  id={w["id"]}  {w["name"]}')
```

### Inspect Workflow Node Inventory (Python)

```python
req = urllib.request.Request(f'{API_BASE}/workflows/{WF_ID}', headers=headers)
with urllib.request.urlopen(req) as r:
    wf = json.loads(r.read())
print(f'Name: {wf["name"]}  Active: {wf["active"]}  Nodes: {len(wf["nodes"])}')
for n in wf['nodes']:
    creds = list(n.get('credentials', {}).keys())
    print(f'  [{n["type"]} v{n["typeVersion"]}] {n["name"]}' + (f'  creds={creds}' if creds else ''))
```

---

## Testing Workflows

### Pre-Deploy Validation (Local)

Before deploying, validate JSON structure locally:

```python
import json, sys

with open('workflows/my-workflow.workflow.json', encoding='utf-8') as f:
    wf = json.load(f)

errors = []
node_names = {n['name'] for n in wf['nodes']}
conn_keys   = set(wf.get('connections', {}).keys())

# Check connection keys match node names
for key in conn_keys:
    if key not in node_names:
        errors.append(f'Connection key "{key}" has no matching node')

# Check for unicode escapes
raw = open('workflows/my-workflow.workflow.json', encoding='utf-8').read()
if r'\u0' in raw or r'\u04' in raw:
    errors.append('Unicode escapes found — save with ensure_ascii=False')

# Check settings
if wf.get('settings', {}).get('executionOrder') != 'v1':
    errors.append('settings.executionOrder must be "v1"')

print('Errors:', errors if errors else 'None — OK')
print(f'Nodes: {len(wf["nodes"])}  Connections: {len(conn_keys)}')
```

### Test on n8n Host (Trigger Manual Run + Watch)

```powershell
# 1. Deploy current local file
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json

# 2. Trigger a test execution
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Execute

# 3. Watch logs for execution output
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 2>&1 | tail -30"

# 4. Check execution result
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions -ExecutionsLimit 3
```

### Full Deploy + Test Cycle

```powershell
# Deploy, activate, trigger, and check status in one go
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Activate
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Execute
Start-Sleep -Seconds 5
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions -ExecutionsLimit 3
```
