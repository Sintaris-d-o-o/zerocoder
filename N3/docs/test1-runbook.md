# Test1 Runbook

## Files
- `workflows/test1-content-factory.workflow.json` — n8n workflow for the content factory
- `scripts/publish-workflow.ps1` — updates workflow in n8n by `N8N_WORKFLOW_ID` from `.env`

## Publish
Run from repo root:

```powershell
pwsh -File .\scripts\publish-workflow.ps1
```

Publish and activate:

```powershell
pwsh -File .\scripts\publish-workflow.ps1 -Activate
```

## After import/update in n8n
1. Open workflow and attach Google Sheets OAuth credentials to:
   - `Google Sheets Trigger`
   - `Update Row Processing`
   - `Update Row Done`
2. Save and activate workflow.
3. Add row in sheet with:
   - `source_url` = article URL
   - `status` = `new`
4. Verify row gets `processing` then `done` and `post_text` is filled.
