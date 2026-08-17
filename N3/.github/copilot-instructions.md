# Zerocoder — n8n Workflow Project

This workspace is for building and deploying n8n automation workflows to a remote n8n instance.

## Skills & Instructions — MANDATORY

**BLOCKING REQUIREMENT**: Before working on ANY n8n-related task, you MUST load and read the skill file FIRST:

```
.github/skills/n8n-workflow/SKILL.md
```

This covers: building workflows, editing nodes, deploying, debugging triggers, checking VPS status, reading Docker logs, diagnosing Gmail rate limits, monitoring executions, activating/deactivating workflows, manually executing workflows, listing/inspecting executions and errors, reviewing live workflow structure, comparing local vs live, listing all workflows, testing workflows on the n8n host.

When editing `workflows/*.workflow.json`, the instruction `.github/instructions/n8n-workflow-json.instructions.md` also applies automatically.

When editing `scripts/` or performing operational tasks (status checks, log reads, deploys), load `.github/instructions/n8n-ops.instructions.md`.

**Never proceed with an n8n task without loading the relevant skill/instruction first.**

---

## Environment

- **n8n host / API base / API key**: see `.env` → `N8N_HOST`, `N8N_API_BASE`, `N8N_API_KEY`
- **Universal deploy script**: `scripts/publish.ps1` — works for any workflow
- **Workflow files**: `workflows/*.workflow.json`
- **Workflow IDs**: stored in `.env` as `N8N_WF_<SLUG>_ID` (auto-derived from filename)

## VPS & Docker

| Resource | Value |
|---|---|
| SSH | `ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de` |
| Container name | `n8n-docker-n8n-1` (NOT `n8n_docker`) |
| n8n version | `1.113.3` |
| n8n API base | `https://automata.dev2null.de/api/v1` |

```bash
# Quick status check
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker ps | grep n8n && date"

# View logs
ssh -i ~/.ssh/id_ed25519 -p 22 stas@dev2null.de "sudo docker logs n8n-docker-n8n-1 2>&1 | tail -50"
```

## Deploy Any Workflow

Always use `scripts/publish.ps1` — never manually PUT workflows:
```powershell
# Deploy and update
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json

# Create new workflow in n8n (auto-saves ID to .env)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Create

# Deploy + activate
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Activate

# Check status / executions
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Status

# List executions with error details
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions -ExecutionsLimit 10

# Trigger manual test run on VPS (bypasses trigger)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Execute

# Review live node inventory (compare with local)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Review

# List all workflows on n8n
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -ListAll

# Deactivate
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Deactivate

# Delete (with confirmation)
.\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Delete
```

Workflow-specific wrappers (kept for backward compatibility):
- `scripts/publish-hr-workflow.ps1` — HR assistant
- `scripts/publish-workflow.ps1` — Content factory

## Key Conventions

- Workflow JSON files live in `workflows/` folder, named `*.workflow.json`
- `.env` stores all credentials and IDs — never hardcode them
- Workflow IDs auto-derived: filename slug → `N8N_WF_<SLUG>_ID` (e.g. `my-flow.workflow.json` → `N8N_WF_MY_FLOW_ID`)
- Use `python3` for API calls in bash scripts (curl + python3 pipes)
- Use PowerShell `Invoke-RestMethod` in `.ps1` scripts

## Node Compatibility

For the full verified node compatibility table, see `.github/skills/n8n-workflow/SKILL.md`.
**Critical**: Use `@n8n/n8n-nodes-langchain.openAi` (v2.1), NOT `n8n-nodes-base.openAi` (broken).

## Credentials (already configured in n8n)

All credential IDs are in `.env` → `N8N_*_CREDENTIAL_ID` vars. Never hardcode them in workflow JSON or docs.

| credentialType | `.env` var |
|---|---|
| `openAiApi` | `N8N_OPENAI_CREDENTIAL_ID` |
| `googleSheetsOAuth2Api` | `N8N_GSHEETS_CREDENTIAL_ID` |
| `googleSheetsTriggerOAuth2Api` | `N8N_GSHEETS_TRIGGER_CREDENTIAL_ID` |
| `gmailOAuth2` | `N8N_GMAIL_CREDENTIAL_ID` |

New credential IDs: add to `.env` as `N8N_<SERVICE>_CREDENTIAL_ID` and reference from there.
