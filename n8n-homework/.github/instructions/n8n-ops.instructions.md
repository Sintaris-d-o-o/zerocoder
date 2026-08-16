---
description: 'Use for n8n operational tasks: checking VPS status, reading Docker logs, calling the n8n REST API, deactivating/activating workflows, diagnosing Gmail rate limits, deploying via publish scripts, manually executing workflows, listing/inspecting executions, reviewing live workflow structure, listing all workflows, testing workflows on n8n host. Load this whenever the user asks to: check n8n status, view logs, SSH to VPS, debug trigger issues, activate/deactivate a workflow, execute a workflow, check executions, review a workflow, list workflows.'
applyTo: "workflows/**,scripts/**"
---

# n8n Operations Reference

## VPS Connection

```bash
# Container status + server time
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker ps | grep n8n && date"

# Latest logs (last 50 lines)
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 2>&1 | tail -50"

# Logs from last N hours
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 --since 2h 2>&1 | tail -80"

# Filter logs for a specific workflow ID or keyword
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 2>&1 | grep '<WORKFLOW_ID>' | tail -20"
```

**Container name**: `n8n-docker-n8n-1` — NEVER `n8n_docker` (old name, does not exist).

## API Patterns

Read credentials from `.env` — NEVER hardcode API key or workflow IDs.

```python
import urllib.request, json

# Load .env
env = {}
with open('.env', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip()

API_BASE = env['N8N_API_BASE'].rstrip('/') + '/v1'
API_KEY  = env['N8N_API_KEY']
WF_ID    = env['N8N_WF_MY_WORKFLOW_ID']  # replace with actual key from .env
headers  = {'X-N8N-API-KEY': API_KEY}
```

### Workflow Status
```python
req = urllib.request.Request(f'{API_BASE}/workflows/{WF_ID}', headers=headers)
with urllib.request.urlopen(req) as r:
    wf = json.loads(r.read())
print(f'active={wf["active"]}  name={wf["name"]}  nodes={len(wf["nodes"])}')
```

### List All Workflows
```python
req = urllib.request.Request(f'{API_BASE}/workflows?limit=100', headers=headers)
with urllib.request.urlopen(req) as r:
    wfs = json.loads(r.read())
for w in wfs['data']:
    print(f'{"[ACTIVE]" if w["active"] else "[off]   "}  id={w["id"]}  {w["name"]}')
```

### List Recent Executions
```python
# All statuses
req = urllib.request.Request(f'{API_BASE}/executions?workflowId={WF_ID}&limit=10', headers=headers)
with urllib.request.urlopen(req) as r:
    execs = json.loads(r.read())
for e in execs['data']:
    print(f'  id={e["id"]} status={e["status"]} mode={e["mode"]} startedAt={e["startedAt"]}')

# Only errors
req = urllib.request.Request(f'{API_BASE}/executions?workflowId={WF_ID}&status=error&limit=5', headers=headers)
with urllib.request.urlopen(req) as r:
    errs = json.loads(r.read())
print(f'Failed: {len(errs["data"])}')
```

### Get Execution Details (Node Output / Error)
```python
EXEC_ID = '123'
req = urllib.request.Request(f'{API_BASE}/executions/{EXEC_ID}', headers=headers)
with urllib.request.urlopen(req) as r:
    detail = json.loads(r.read())

# Error info
err = detail['data']['resultData'].get('error', {})
if err:
    print('Error:', err.get('message'))
    print('Node :', err.get('node', {}).get('name'))

# Node output data
for node_name, runs in detail['data']['resultData'].get('runData', {}).items():
    for run in runs:
        items = run.get('data', {}).get('main', [[]])[0]
        print(f'{node_name}: {len(items)} items')
```

### Manually Execute Workflow (SSH + docker exec)
```bash
# Run workflow manually on VPS (bypasses trigger, starts from first node)
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de \
  "sudo docker exec n8n-docker-n8n-1 n8n execute --id=<WORKFLOW_ID> 2>&1 | tail -30"
```

### Deactivate / Activate
```python
h2 = {**headers, 'Content-Type': 'application/json'}

req = urllib.request.Request(f'{API_BASE}/workflows/{WF_ID}/deactivate', data=b'{}', headers=h2, method='POST')
with urllib.request.urlopen(req) as r:
    print('Deactivated:', json.loads(r.read()).get('active'))

req = urllib.request.Request(f'{API_BASE}/workflows/{WF_ID}/activate', data=b'{}', headers=h2, method='POST')
with urllib.request.urlopen(req) as r:
    print('Activated:', json.loads(r.read()).get('active'))
```

## Gmail Rate Limit Diagnosis

**Log pattern**: `User-rate limit exceeded. Retry after 2026-04-06T07:40:23.781Z`

- Google's rate limit is a **sliding window**: every failed poll attempt resets the timer. n8n does NOT honor `Retry-After` and keeps polling on its fixed schedule — so the window can never expire while the workflow is active
- **Immediate fix: deactivate the workflow** to stop all polling (logs will show `Deregistered all crons for workflow`)
- Wait **15–30 minutes**, then reactivate: `.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Activate`
- Do NOT restart the container — that causes a new startup burst and resets the window again

## Deploy / Manage via Universal Publish Script

```powershell
# Update any workflow
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json

# Update + activate
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Activate

# Create new workflow in n8n
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Create

# Check status + last 5 executions
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Status

# List last 10 executions with error detail
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Executions

# List last N executions
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Executions -ExecutionsLimit 20

# Manually execute on VPS (test run, bypasses trigger)
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Execute

# Review live structure + compare with local
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Review

# List all workflows on the n8n instance
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -ListAll

# Deactivate
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Deactivate

# Delete (with confirmation)
.\scripts\publish.ps1 -WorkflowFile workflows/<name>.workflow.json -Delete
```

The script auto-derives the `.env` ID key from the filename:
`my-workflow.workflow.json` → `N8N_WF_MY_WORKFLOW_ID`

Legacy wrappers for existing workflows:
- `scripts/publish-hr-workflow.ps1` — HR assistant
- `scripts/publish-workflow.ps1` — Content factory
