---
description: 'Use when editing, creating, or reviewing n8n workflow JSON files (*.workflow.json). Enforces correct node types, typeVersions, credential IDs, and connection formats. Credential IDs and URLs are in .env.'
applyTo: "workflows/*.workflow.json"
---

# n8n Workflow JSON Rules

## Required structure

Every workflow JSON must have:
```json
{
  "name": "...",
  "settings": { "executionOrder": "v1" },
  "nodes": [...],
  "connections": { ... }
}
```

## OpenAI node — ONLY valid form

```json
{
  "type": "@n8n/n8n-nodes-langchain.openAi",
  "typeVersion": 2.1,
  "credentials": { "openAiApi": { "id": "<N8N_OPENAI_CREDENTIAL_ID>", "name": "<N8N_OPENAI_CREDENTIAL_NAME>" } },
  "parameters": {
    "modelId": { "__rl": true, "value": "gpt-4o", "mode": "list", "cachedResultName": "gpt-4o" },
    "responses": { "values": [ { "role": "system", "content": "..." }, { "content": "=..." } ] },
    "options": { "temperature": 0.3 }
  }
}
```

- **WRONG**: `n8n-nodes-base.openAi` — does not exist in this instance
- **WRONG**: `responses` key named `messages` — wrong parameter name in v2.1
- **Output**: always at `$json.content`, NOT `$json.choices[0].message.content`

## Google Sheets credential

All credential IDs come from `.env` — see `N8N_GSHEETS_CREDENTIAL_ID`, `N8N_GSHEETS_TRIGGER_CREDENTIAL_ID`, `N8N_GMAIL_CREDENTIAL_ID`.

```json
"credentials": {
  "googleSheetsOAuth2Api": { "id": "<N8N_GSHEETS_CREDENTIAL_ID>", "name": "<N8N_GSHEETS_CREDENTIAL_NAME>" }
}
```

Trigger credential: `"googleSheetsTriggerOAuth2Api": { "id": "<N8N_GSHEETS_TRIGGER_CREDENTIAL_ID>", "name": "<N8N_GSHEETS_TRIGGER_CREDENTIAL_NAME>" }`

## Verified typeVersions

| type | typeVersion |
|---|---|
| `n8n-nodes-base.code` | `2` |
| `n8n-nodes-base.googleSheets` | `4.5` |
| `n8n-nodes-base.httpRequest` | `4.2` |
| `n8n-nodes-base.gmail` | `2.1` |
| `n8n-nodes-base.gmailTrigger` | `1` |
| `n8n-nodes-base.wait` | `1.1` |
| `n8n-nodes-base.if` | `1` |
| `n8n-nodes-base.extractFromFile` | `1` |
| `@n8n/n8n-nodes-langchain.openAi` | `2.1` |

## Deploy

Strip `id`, `active`, `createdAt`, `updatedAt`, `versionId` before POST or PUT.  
Use `scripts/publish.ps1 -WorkflowFile workflows/<name>.workflow.json` for any workflow.

## Quality Review (before publishing)

- Every node has an incoming **and** outgoing connection (except trigger/leaf nodes)
- Connection object keys exactly match node `"name"` values
- No two Code nodes connected directly in sequence — merge them
- No `\uXXXX` unicode escapes in strings — save JSON with `ensure_ascii=False`
- System prompts live in the OpenAI node `responses.values`, not in Code node variables
- All `$('Node Name')` references in jsCode match current node names
- Credential IDs come from `.env` `N8N_*_CREDENTIAL_ID` vars

## Local Validation (before deploy)

Quick structure check from terminal:
```python
import json
with open('workflows/my-workflow.workflow.json', encoding='utf-8') as f:
    wf = json.load(f)
node_names = {n['name'] for n in wf['nodes']}
conn_keys  = set(wf.get('connections', {}).keys())
bad_keys   = conn_keys - node_names
print('Bad connection keys:', bad_keys or 'none')
print('executionOrder:', wf.get('settings', {}).get('executionOrder'))
print('Nodes:', len(wf['nodes']), '| Connections from:', len(conn_keys))
```

## Test on n8n Host

```powershell
# 1. Deploy
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json

# 2. Trigger manual run (bypasses trigger node, starts from node 1)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Execute

# 3. Check result
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions -ExecutionsLimit 3

# 4. Review node inventory (live vs. local comparison)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Review
```
