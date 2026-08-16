# Copilot Customization User Guide

This guide explains how GitHub Copilot is configured in this project — what files exist, how they work together, and how Copilot uses them when you ask a question or give it a task.

---

## Overview: 4 Types of Customization Files

```
.github/
├── copilot-instructions.md          ← Global rules (always active)
├── instructions/
│   ├── n8n-workflow-json.instructions.md   ← Auto-loads for workflow JSON files
│   └── n8n-ops.instructions.md             ← Auto-loads for scripts/ and workflows/
├── skills/
│   └── n8n-workflow/
│       └── SKILL.md                 ← Deep expertise, loaded on demand
└── agents/
    └── n8n-publisher.agent.md       ← Specialized sub-agent for deployments
```

Each type has a different **trigger** and **scope**:

| File type | When it loads | Scope | Purpose |
|---|---|---|---|
| `copilot-instructions.md` | Always, every request | Entire workspace | Global rules & project overview |
| `instructions/*.instructions.md` | Auto, based on `applyTo` glob | Specific files/folders | Enforce rules when editing specific files |
| `skills/*/SKILL.md` | On demand — Copilot reads it before acting | Any topic | Deep reference knowledge + verified patterns |
| `agents/*.agent.md` | When you pick an agent mode | Scoped session | Specialized autonomous assistant |

**Script entry point for everything**: `scripts/publish.ps1` is the universal tool for all workflow lifecycle operations — deploy, create, activate, deactivate, execute, review, list executions, list all workflows.

---

## Flow Diagram

```
You type a request in Copilot Chat
            │
            ▼
  ┌─────────────────────────────────────┐
  │  copilot-instructions.md            │  ← ALWAYS loaded first
  │  (global rules, project context,    │
  │   mandatory skill-loading rules,    │
  │   VPS quick reference)              │
  └─────────────────────────────────────┘
            │
            ├──── Are you editing a workflow JSON? ──────────────────────────────►
            │                                                                      │
            │         Auto-loads:                                                  │
            │         n8n-workflow-json.instructions.md                            │
            │         (node types, typeVersions, credential format, deploy rules)  │
            │                                                                      ▼
            │                                                            Copilot enforces
            │                                                            correct JSON structure
            │
            ├──── Are you editing scripts/ or asking about ops? ────────────────►
            │                                                                      │
            │         Auto-loads:                                                  │
            │         n8n-ops.instructions.md                                      │
            │         (SSH commands, API patterns, Gmail rate limit guide)         │
            │                                                                      ▼
            │                                                            Copilot uses
            │                                                            correct ops procedures
            │
            ├──── Is the task n8n-related? (build, debug, deploy, check status,  ─►
            │     execute, review, list executions, list workflows, test)           │
            │                                                                      │
            │         Copilot MUST read first:                                     │
            │         .github/skills/n8n-workflow/SKILL.md                         │
            │         (verified nodes, VPS ops, API patterns, known bugs,          │
            │          deploy procedure, execution management, review & testing,   │
            │          Gmail rate limit diagnosis)                                  │
            │                                                                      ▼
            │                                                            Copilot acts with
            │                                                            deep project knowledge
            │
            └──── Did you select "n8n Publisher" agent mode? ──────────────────►
                                                                                   │
                        Loads: n8n-publisher.agent.md                              │
                        (deployment specialist: PUT/POST workflow,                 │
                         activate, deactivate, execute, review, list              │
                         executions, save IDs to .env)                            │
                                                                                   ▼
                                                                         Autonomous deploy
                                                                         agent runs
```

---

## Each File Explained

### 1. `copilot-instructions.md` — Global Rules

**Location**: `.github/copilot-instructions.md`  
**Loads**: Every single request, automatically.  
**Purpose**: The "always on" rulebook for this project.

Contains:
- **BLOCKING REQUIREMENT**: Copilot must load `SKILL.md` before any n8n work
- Project overview (what this repo is for)
- VPS quick reference: SSH command, correct Docker container name, n8n API base
- Credential table: which `.env` variable maps to which `credentialType`
- Deploy script usage
- Key conventions (no hardcoded credentials, python3 for bash scripts, etc.)

**Why it exists**: Without it, Copilot would use the wrong Docker container name (`n8n_docker` instead of `n8n-docker-n8n-1`), hardcode credentials, or try to use broken node types.

---

### 2. `instructions/n8n-workflow-json.instructions.md` — JSON Editing Rules

**Location**: `.github/instructions/n8n-workflow-json.instructions.md`  
**Loads**: Automatically when you open or edit any file matching `workflows/*.workflow.json`  
**Purpose**: Enforce correct n8n JSON structure without you having to ask.

Contains:
- Required top-level fields (`name`, `settings`, `nodes`, `connections`)
- The ONLY valid OpenAI node form (`@n8n/n8n-nodes-langchain.openAi` v2.1 with `responses.values`)
- Why `n8n-nodes-base.openAi` is broken (doesn't exist in this instance)
- Correct credential block format per node type
- Verified `typeVersion` table for every node used in this project
- Pre-deploy checklist (strip `id`, `active`, `versionId` etc.)
- Quality review rules (all nodes connected, no unicode escapes, credential IDs from `.env`)
- **Local validation snippet** — Python script to check connection keys, unicode escapes, executionOrder before deploying
- **Test on n8n host** pattern: deploy → `-Execute` → `-Executions` → `-Review` cycle

**Why it exists**: n8n's API returns 400 if you send wrong field names or wrong typeVersions. This instruction makes Copilot generate deployable JSON on the first try.

---

### 3. `instructions/n8n-ops.instructions.md` — Operations Reference

**Location**: `.github/instructions/n8n-ops.instructions.md`  
**Loads**: Automatically when editing files in `workflows/**` or `scripts/**`  
**Purpose**: Give Copilot the exact commands for VPS/API operations.

Contains:
- SSH commands with correct container name and flags
- Full Python snippets for: reading `.env`, checking workflow status, listing all workflows, listing executions, getting execution details (node output + error), manually executing via `docker exec`, deactivating/activating
- Gmail rate limit diagnosis: sliding window explanation, what NOT to do (don't restart container), recovery procedure (deactivate → wait 15–30 min → reactivate)
- Full publish script command reference: all flags including `-Execute`, `-Executions`, `-Review`, `-ListAll`

**Why it exists**: Gaps between sessions mean Copilot may forget the correct container name or use `curl` instead of `python3`. This file provides ready-to-run patterns.

---

### 4. `skills/n8n-workflow/SKILL.md` — Deep Knowledge Base

**Location**: `.github/skills/n8n-workflow/SKILL.md`  
**Loads**: On demand — Copilot reads this file when an n8n task is detected.  
**Purpose**: The single source of truth for all n8n knowledge in this project.

Contains all sections from the instructions files PLUS:
- Full node compatibility table (all verified `type` + `typeVersion` combinations)  
- Known broken nodes (`n8n-nodes-base.openAi` → use langchain version)  
- Complete node JSON structures (OpenAI, Google Sheets, Gmail, Code node patterns)  
- Workflow JSON structure and connections format  
- Deploy procedure (PowerShell scripts + manual Python API method)  
- VPS & operations section: SSH, Docker logs, API status/executions/activate scripts  
- Gmail rate limit root cause, recovery steps, prevention (`everyX` format vs `everyMinute`)  
- **Execution Management**: list executions, get execution detail (node output / error message), manual trigger via `docker exec`  
- **Workflow Review & Inspection**: live node inventory, local vs live comparison, list all workflows  
- **Testing Workflows**: local validation script, full deploy+test cycle pattern  
- Quality review checklist (9 items)  
- Common mistakes to avoid (6 items)

**Why it exists**: Instructions files are short rule-sets. The skill is a complete reference manual. When Copilot needs to build a workflow from scratch or debug a complex issue, it reads the skill first and has everything it needs without web searches.

---

### 5. `agents/n8n-publisher.agent.md` — Deployment & Operations Agent

**Location**: `.github/agents/n8n-publisher.agent.md`  
**Loads**: When you switch to "n8n Publisher" agent mode in Copilot Chat.  
**Purpose**: A specialized autonomous agent for the full n8n workflow lifecycle — deploy, execute, monitor, inspect.

Contains:
- Role definition: deployment & operations specialist
- Allowed tools: `read`, `edit`, `execute`, `search`
- Instance config (API base, auth header name, SSH, Docker container)
- Deployment rules (strip read-only fields, never send `active` on POST, etc.)
- Full script command reference (all publish.ps1 flags)
- Execution monitoring rules: show status distribution, fetch error detail for failed executions
- Testing workflow: deploy → `-Execute` → wait → `-Executions` → inspect errors if any
- Verification step: after every deploy, fetch and print node types to confirm no broken nodes

**How to activate**: In Copilot Chat, click the agent picker and choose **"n8n Publisher"**. Then say things like:
- "publish the `my-workflow.workflow.json` workflow"
- "create a new workflow from `ideas.workflow.json`"
- "show me the last 10 executions of the HR workflow"
- "test the content factory workflow on the VPS"
- "review what nodes are live on n8n vs. local"
- "list all workflows"

**Why it exists**: Separates deployment concerns from authoring. The agent knows exactly which fields to strip, which endpoint to call (POST vs PUT), how to test a run end-to-end, and always verifies the result — without you having to explain every time.

---

## How Files Interact — Example Scenarios

### Scenario A: "Add a new node to any workflow"

1. `copilot-instructions.md` loads → Copilot reads mandatory-skill rule
2. Copilot reads `SKILL.md` → loads verified node table, credential format
3. You open any `*.workflow.json` → `n8n-workflow-json.instructions.md` auto-loads
4. Copilot edits the file using correct `type`/`typeVersion`, credentials from `.env`, valid connections format

### Scenario B: "Check why the Gmail trigger isn't firing"

1. `copilot-instructions.md` loads → VPS quick reference available
2. Copilot reads `SKILL.md` → Gmail rate limit section loaded
3. `n8n-ops.instructions.md` auto-loads (workflow files are open)
4. Copilot SSHs with correct container name, reads logs, identifies `User-rate limit exceeded`, advises waiting — does NOT restart container

### Scenario C: "Deploy a workflow"

**Option 1 — default Copilot Chat:**
1. `copilot-instructions.md` → mandatory skill rule
2. Copilot reads `SKILL.md` → deploy procedure section
3. Copilot runs `.\.scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Activate`

**Option 2 — n8n Publisher agent:**
1. You pick **n8n Publisher** agent mode
2. `n8n-publisher.agent.md` loads → agent takes full ownership
3. Agent lists `workflows/*.workflow.json`, reads `.env` for existing IDs, strips meta fields, calls PUT or POST, verifies result, reports

### Scenario D: "Edit a publish script or scripts/"

1. `copilot-instructions.md` loads
2. File is in `scripts/` → `n8n-ops.instructions.md` auto-loads
3. Copilot knows to use `encoding='utf-8'` when reading `.env` (Windows UTF-8 requirement), uses `python3` not `curl`

### Scenario E: "Check what happened in last executions / why did it fail?"

1. `copilot-instructions.md` → SKILL.md must be read
2. Copilot reads `SKILL.md` → Execution Management section
3. `n8n-ops.instructions.md` is available (workflow files open) with Python execution API snippets
4. Copilot runs: `.\.scripts\publish.ps1 -Executions -ExecutionsLimit 10`
5. For any `error` execution, fetches detail → shows error message + failing node name

### Scenario F: "Test the workflow on the VPS"

1. `copilot-instructions.md` → skill rule
2. Copilot reads `SKILL.md` → Testing Workflows section
3. Full test cycle: `publish.ps1` (deploy) → `-Execute` (trigger) → wait 5s → `-Executions -ExecutionsLimit 3` (check result) → `-Review` (compare local vs live if needed)

### Scenario G: "What workflows are running on n8n?"

1. `copilot-instructions.md` → skill rule
2. Copilot reads `SKILL.md` → Workflow Review section
3. Runs `.\.scripts\publish.ps1 -WorkflowFile workflows/any.workflow.json -ListAll`
4. Shows all workflows with active/inactive state and IDs

---

## File Locations Summary

```
.github/
├── copilot-instructions.md
│     Always active. Global rules, VPS reference, mandatory skill-loading.
│     Deploy script reference including -Execute, -Review, -Executions, -ListAll.
│
├── instructions/
│   ├── n8n-workflow-json.instructions.md
│   │     applyTo: workflows/*.workflow.json
│   │     Correct node JSON structure, typeVersions, credential format,
│   │     local validation script, test-on-host pattern.
│   │
│   └── n8n-ops.instructions.md
│         applyTo: workflows/**, scripts/**
│         SSH commands, API snippets, list all workflows, list/inspect executions,
│         manual docker exec, Gmail rate limit guide, full publish.ps1 reference.
│
├── skills/
│   └── n8n-workflow/
│       └── SKILL.md
│             On-demand deep reference. Node table, VPS ops, deploy procedure,
│             execution management, review & inspection, testing patterns,
│             known bugs, quality checklist, Gmail rate limit root cause.
│
└── agents/
    └── n8n-publisher.agent.md
          Activated manually. Autonomous deployment & operations specialist.
          POST/PUT workflows, activate, deactivate, execute, review,
          list executions with error detail, list all workflows, save IDs.
```

---

## Adding New Knowledge

When you discover something new (a new working node, a new API quirk, a new VPS procedure):

| What to update | Where |
|---|---|
| New verified node type/version | `SKILL.md` → Verified Compatible Nodes table |
| New broken node | `SKILL.md` → Known Broken / Incompatible Nodes |
| New VPS/SSH procedure | `SKILL.md` → VPS & n8n Operations + `n8n-ops.instructions.md` |
| New JSON structure rule | `n8n-workflow-json.instructions.md` |
| New execution/testing pattern | `SKILL.md` → Execution Management / Testing Workflows + `n8n-ops.instructions.md` |
| Credential ID change | `.env` only — never in the docs |
| New workflow ID | `.env` as `N8N_WF_<SLUG>_ID` (auto-saved by `publish.ps1 -Create`) |
| New publish.ps1 flag | `scripts/publish.ps1` + update `SKILL.md`, `n8n-ops.instructions.md`, `copilot-instructions.md` deploy section |
