<#
.SYNOPSIS
  Universal n8n workflow publish script. Works for any workflow in the workflows/ folder.

.PARAMETER WorkflowFile
  Path to the workflow JSON file, relative to the project root.
  Example: "workflows/my-new-workflow.workflow.json"

.PARAMETER EnvKey
  Name of the .env variable that stores this workflow's n8n ID.
  Auto-derived from WorkflowFile name if omitted.
  Derivation rule: filename slug uppercased, non-alphanumeric -> underscore, wrapped as N8N_WF_<SLUG>_ID
  Example: "my-new-workflow.workflow.json" -> N8N_WF_MY_NEW_WORKFLOW_ID

.PARAMETER WorkflowId
  Explicit n8n workflow ID. Overrides .env lookup entirely.

.PARAMETER Create
  Force create a new workflow in n8n even if an ID already exists in .env.

.PARAMETER Activate
  Activate the workflow after deploying (or standalone if no deploy needed).

.PARAMETER Deactivate
  Deactivate the workflow without deploying.

.PARAMETER Status
  Show workflow active state and last 5 executions. No deploy.

.PARAMETER Delete
  Delete the workflow from n8n and clear its ID from .env. Requires confirmation.

.PARAMETER Execute
  Trigger a manual workflow execution on the VPS via SSH + docker exec.

.PARAMETER Executions
  Show recent executions. Use -ExecutionsLimit to control how many (default: 10).

.PARAMETER ExecutionsLimit
  Number of executions to show with -Executions. Default: 10.

.PARAMETER Review
  Fetch the live workflow JSON from n8n and display node inventory. Compare node count with local file.

.PARAMETER ListAll
  List all workflows on the n8n instance (name, id, active state).

.EXAMPLE
  # Deploy (update)
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json

  # Deploy and activate
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Activate

  # Create new
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Create

  # Check status
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Status

  # Deactivate only
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Deactivate

  # Delete (with confirmation)
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Delete

  # Execute workflow manually on VPS
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Execute

  # Show last 10 executions
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions

  # Show last 20 executions
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Executions -ExecutionsLimit 20

  # Review live workflow structure on n8n
  .\scripts\publish.ps1 -WorkflowFile workflows/my-workflow.workflow.json -Review

  # List all workflows on n8n instance
  .\scripts\publish.ps1 -WorkflowFile workflows/any.workflow.json -ListAll

  # Use explicit env key (for older workflows with custom .env var names)
  .\scripts\publish.ps1 -WorkflowFile workflows/n3-hr-assistant.workflow.json -EnvKey N8N_HR_WORKFLOW_ID
#>
param(
  [Parameter(Mandatory)]
  [string]$WorkflowFile,

  [string]$EnvKey = "",
  [string]$WorkflowId = "",

  [switch]$Create,
  [switch]$Activate,
  [switch]$Deactivate,
  [switch]$Status,
  [switch]$Delete,
  [switch]$Execute,
  [switch]$Executions,
  [int]$ExecutionsLimit = 10,
  [switch]$Review,
  [switch]$ListAll
)

$ErrorActionPreference = 'Stop'

# ── Helpers ────────────────────────────────────────────────────────────────────

function Parse-EnvFile {
  param([string]$Path)
  if (!(Test-Path -LiteralPath $Path)) { throw ".env file not found at $Path" }
  $map = @{}
  foreach ($line in Get-Content -LiteralPath $Path) {
    $trimmed = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) { continue }
    if ($trimmed.StartsWith('#')) { continue }
    if ($trimmed -notmatch '^[A-Za-z_][A-Za-z0-9_]*=') { continue }
    $parts = $trimmed -split '=', 2
    $map[$parts[0].Trim()] = $parts[1].Trim()
  }
  return $map
}

function Derive-EnvKey {
  param([string]$FilePath)
  $basename = Split-Path $FilePath -Leaf
  $slug = $basename -replace '\.workflow\.json$', ''
  $slug = $slug.ToUpper() -replace '[^A-Z0-9]', '_'
  return "N8N_WF_${slug}_ID"
}

function Set-EnvValue {
  param([string]$Path, [string]$Key, [string]$Value)
  $content = Get-Content -Raw -LiteralPath $Path
  if ($content -match "(?m)^${Key}=") {
    $content = $content -replace "(?m)^${Key}=.*$", "${Key}=${Value}"
  } else {
    $content = $content.TrimEnd() + "`n${Key}=${Value}`n"
  }
  Set-Content -LiteralPath $Path -Value $content -NoNewline
}

# ── Resolve paths ──────────────────────────────────────────────────────────────

$root      = Resolve-Path (Join-Path $PSScriptRoot '..')
$envPath   = Join-Path $root '.env'
$envVars   = Parse-EnvFile -Path $envPath
$wfAbsPath = Join-Path $root $WorkflowFile

if (!(Test-Path -LiteralPath $wfAbsPath)) {
  throw "Workflow file not found: $wfAbsPath"
}

# ── Resolve env key & workflow ID ─────────────────────────────────────────────

if ([string]::IsNullOrWhiteSpace($EnvKey)) {
  $EnvKey = Derive-EnvKey -FilePath $WorkflowFile
}

if ([string]::IsNullOrWhiteSpace($WorkflowId)) {
  $WorkflowId = if ($envVars.ContainsKey($EnvKey)) { $envVars[$EnvKey] } else { "" }
}

$isNew = [string]::IsNullOrWhiteSpace($WorkflowId) -or $Create.IsPresent

Write-Host "Workflow file : $WorkflowFile"
Write-Host "Env key       : $EnvKey"
Write-Host "Workflow ID   : $(if ($WorkflowId) { $WorkflowId } else { '<none — will create>' })"

# ── Validate required .env vars ────────────────────────────────────────────────

foreach ($key in @('N8N_API_BASE', 'N8N_API_KEY')) {
  if (-not $envVars.ContainsKey($key) -or [string]::IsNullOrWhiteSpace($envVars[$key])) {
    throw "Missing required .env variable: $key"
  }
}

$apiBase = $envVars['N8N_API_BASE'].TrimEnd('/')
$headers = @{
  'X-N8N-API-KEY' = $envVars['N8N_API_KEY']
  'Content-Type'  = 'application/json'
}

# ── LIST ALL ──────────────────────────────────────────────────────────────────

if ($ListAll.IsPresent) {
  $resp = Invoke-RestMethod -Method Get -Uri "$apiBase/v1/workflows?limit=100" -Headers $headers
  Write-Host ""
  Write-Host "=== All Workflows on n8n ==="
  foreach ($w in $resp.data) {
    $activeTag = if ($w.active) { "[ACTIVE]" } else { "[off]   " }
    Write-Host "  $activeTag  id=$($w.id)  $($w.name)"
  }
  Write-Host ""
  Write-Host "Total: $($resp.data.Count)"
  exit 0
}

# ── STATUS ────────────────────────────────────────────────────────────────────

if ($Status.IsPresent) {
  if ([string]::IsNullOrWhiteSpace($WorkflowId)) {
    throw "Cannot check status: no workflow ID found for env key '$EnvKey'"
  }
  $wf       = Invoke-RestMethod -Method Get -Uri "$apiBase/v1/workflows/$WorkflowId" -Headers $headers
  $execResp = Invoke-RestMethod -Method Get -Uri "$apiBase/v1/executions?workflowId=$WorkflowId&limit=5" -Headers $headers
  Write-Host ""
  Write-Host "=== Workflow Status ==="
  Write-Host "Name   : $($wf.name)"
  Write-Host "ID     : $($wf.id)"
  Write-Host "Active : $($wf.active)"
  Write-Host ""
  Write-Host "=== Recent Executions ==="
  foreach ($e in $execResp.data) {
    Write-Host "  id=$($e.id)  status=$($e.status)  mode=$($e.mode)  startedAt=$($e.startedAt)"
  }
  exit 0
}

# ── EXECUTIONS ────────────────────────────────────────────────────────────────

if ($Executions.IsPresent) {
  if ([string]::IsNullOrWhiteSpace($WorkflowId)) {
    throw "Cannot list executions: no workflow ID found for env key '$EnvKey'"
  }
  $execResp = Invoke-RestMethod -Method Get -Uri "$apiBase/v1/executions?workflowId=$WorkflowId&limit=$ExecutionsLimit" -Headers $headers
  Write-Host ""
  Write-Host "=== Last $ExecutionsLimit Executions ==="
  Write-Host "Workflow: $(if ($execResp.data.Count -gt 0) { $execResp.data[0].workflowId } else { $WorkflowId })"
  Write-Host ""
  foreach ($e in $execResp.data) {
    $dur = if ($e.stoppedAt -and $e.startedAt) {
      try { "$([math]::Round(([datetime]$e.stoppedAt - [datetime]$e.startedAt).TotalSeconds, 1))s" }
      catch { "?" }
    } else { "?" }
    Write-Host "  id=$($e.id)  status=$($e.status)  mode=$($e.mode)  dur=$dur  startedAt=$($e.startedAt)"
  }
  if ($execResp.data.Count -eq 0) { Write-Host "  (no executions found)" }
  Write-Host ""

  # Fetch details of the last failed execution if any
  $lastError = $execResp.data | Where-Object { $_.status -eq 'error' } | Select-Object -First 1
  if ($lastError) {
    Write-Host "--- Last Error Detail (id=$($lastError.id)) ---"
    $detail = Invoke-RestMethod -Method Get -Uri "$apiBase/v1/executions/$($lastError.id)" -Headers $headers
    try {
      $errMsg = $detail.data.resultData.error.message
      if ($errMsg) { Write-Host "  Error: $errMsg" }
      $nodeName = $detail.data.resultData.error.node.name
      if ($nodeName) { Write-Host "  Node : $nodeName" }
    } catch {
      Write-Host "  (could not parse error detail)"
    }
  }
  exit 0
}

# ── EXECUTE (manual trigger via SSH + docker exec) ────────────────────────────

if ($Execute.IsPresent) {
  if ([string]::IsNullOrWhiteSpace($WorkflowId)) {
    throw "Cannot execute: no workflow ID found for env key '$EnvKey'"
  }
  $sshUser = if ($envVars.ContainsKey('SSH_USER')) { $envVars['SSH_USER'] } else { 'stas' }
  $sshHost = if ($envVars.ContainsKey('SSH_HOST')) { $envVars['SSH_HOST'] } else { 'dev2null.de' }
  $sshPort = if ($envVars.ContainsKey('SSH_PORT')) { $envVars['SSH_PORT'] } else { '22' }
  $sshKey  = if ($envVars.ContainsKey('SSH_KEY_PATH')) { $envVars['SSH_KEY_PATH'] -replace '^~', $HOME } else { "$HOME/.ssh/id_ed25519" }

  Write-Host "Triggering workflow $WorkflowId on VPS ..."
  Write-Host "(SSH: $sshUser@$sshHost -p $sshPort)"
  Write-Host ""
  ssh -i $sshKey -p $sshPort "${sshUser}@${sshHost}" "sudo docker exec n8n-docker-n8n-1 n8n execute --id=$WorkflowId 2>&1 | tail -30"
  exit 0
}

# ── REVIEW (fetch live workflow from n8n + compare with local) ────────────────

if ($Review.IsPresent) {
  if ([string]::IsNullOrWhiteSpace($WorkflowId)) {
    throw "Cannot review: no workflow ID found for env key '$EnvKey'"
  }
  $wf = Invoke-RestMethod -Method Get -Uri "$apiBase/v1/workflows/$WorkflowId" -Headers $headers
  Write-Host ""
  Write-Host "=== Live Workflow: $($wf.name) ==="
  Write-Host "ID       : $($wf.id)"
  Write-Host "Active   : $($wf.active)"
  Write-Host "Nodes    : $($wf.nodes.Count)"
  Write-Host "Updated  : $($wf.updatedAt)"
  Write-Host ""
  Write-Host "--- Node Inventory ---"
  foreach ($n in $wf.nodes) {
    $cred = if ($n.credentials) { " [creds: $(($n.credentials.PSObject.Properties.Name) -join ',')]" } else { "" }
    Write-Host "  [$($n.type) v$($n.typeVersion)] $($n.name)$cred"
  }

  # Compare with local file
  if (Test-Path -LiteralPath $wfAbsPath) {
    $local = Get-Content -Raw -LiteralPath $wfAbsPath | ConvertFrom-Json
    Write-Host ""
    Write-Host "--- Local vs Live Comparison ---"
    Write-Host "  Local nodes : $($local.nodes.Count)"
    Write-Host "  Live nodes  : $($wf.nodes.Count)"
    if ($local.nodes.Count -ne $wf.nodes.Count) {
      Write-Warning "Node count mismatch — local file may be out of sync with n8n"
    } else {
      Write-Host "  Node counts match OK"
    }
    $localNames = $local.nodes | ForEach-Object { $_.name } | Sort-Object
    $liveNames  = $wf.nodes   | ForEach-Object { $_.name } | Sort-Object
    $missing = $localNames | Where-Object { $_ -notin $liveNames }
    $extra   = $liveNames  | Where-Object { $_ -notin $localNames }
    if ($missing) { Write-Warning "Nodes in local but NOT on n8n: $($missing -join ', ')" }
    if ($extra)   { Write-Warning "Nodes on n8n but NOT local   : $($extra -join ', ')" }
    if (-not $missing -and -not $extra) { Write-Host "  Node names match OK" }
  }
  exit 0
}

# ── DEACTIVATE (standalone, no deploy) ────────────────────────────────────────

if ($Deactivate.IsPresent -and -not $Activate.IsPresent -and -not $isNew) {
  if ([string]::IsNullOrWhiteSpace($WorkflowId)) {
    throw "Cannot deactivate: no workflow ID found for env key '$EnvKey'"
  }
  Write-Host "Deactivating $WorkflowId ..."
  $null = Invoke-RestMethod -Method Post -Uri "$apiBase/v1/workflows/$WorkflowId/deactivate" -Headers $headers
  Write-Host "Deactivated."
  exit 0
}

# ── DELETE ────────────────────────────────────────────────────────────────────

if ($Delete.IsPresent) {
  if ([string]::IsNullOrWhiteSpace($WorkflowId)) {
    throw "Cannot delete: no workflow ID found for env key '$EnvKey'"
  }
  $confirm = Read-Host "Type YES to confirm deleting workflow '$WorkflowId' from n8n"
  if ($confirm -ne 'YES') { Write-Host "Cancelled."; exit 0 }
  $null = Invoke-RestMethod -Method Delete -Uri "$apiBase/v1/workflows/$WorkflowId" -Headers $headers
  Set-EnvValue -Path $envPath -Key $EnvKey -Value ""
  Write-Host "Deleted workflow $WorkflowId and cleared $EnvKey in .env"
  exit 0
}

# ── CREATE ────────────────────────────────────────────────────────────────────

if ($isNew) {
  Write-Host "Creating new workflow in n8n ..."
  $payload = Get-Content -Raw -LiteralPath $wfAbsPath | ConvertFrom-Json
  foreach ($field in @('id', 'active', 'createdAt', 'updatedAt', 'versionId', 'meta', 'pinData')) {
    $payload.PSObject.Properties.Remove($field)
  }
  $payload | Add-Member -NotePropertyName 'active' -NotePropertyValue $false -Force

  $resp = Invoke-RestMethod -Method Post -Uri "$apiBase/v1/workflows" -Headers $headers `
    -Body ($payload | ConvertTo-Json -Depth 100)

  $WorkflowId = $resp.id
  Write-Host "Created: $($resp.name) (id=$WorkflowId)"

  Set-EnvValue -Path $envPath -Key $EnvKey -Value $WorkflowId
  Write-Host ".env updated: ${EnvKey}=${WorkflowId}"
}

# ── UPDATE ────────────────────────────────────────────────────────────────────

else {
  Write-Host "Updating workflow $WorkflowId ..."

  # Use Python to preserve UTF-8 encoding (PowerShell ConvertTo-Json can mangle non-ASCII)
  $tmpPy = [System.IO.Path]::GetTempFileName() + ".py"
  Set-Content -LiteralPath $tmpPy -Encoding UTF8 -Value @'
import json, urllib.request, sys
env_path, wf_path, wf_id = sys.argv[1], sys.argv[2], sys.argv[3]
env = {}
with open(env_path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip()
with open(wf_path, encoding='utf-8') as f:
    wf = json.load(f)
# Strip read-only / disallowed fields
for k in ('id', 'active', 'createdAt', 'updatedAt', 'versionId', 'meta', 'pinData'):
    wf.pop(k, None)
body = json.dumps(wf, ensure_ascii=False).encode('utf-8')
req = urllib.request.Request(
    env['N8N_API_BASE'].rstrip('/') + '/v1/workflows/' + wf_id,
    data=body,
    headers={'X-N8N-API-KEY': env['N8N_API_KEY'], 'Content-Type': 'application/json; charset=utf-8'},
    method='PUT'
)
with urllib.request.urlopen(req) as resp:
    r = json.load(resp)
    print('Updated: ' + r['name'] + ' (id=' + r['id'] + ')')
'@

  python3 $tmpPy $envPath $wfAbsPath $WorkflowId
  Remove-Item -LiteralPath $tmpPy -Force

  # Restart polling trigger: deactivate -> activate so n8n re-registers the timer
  Write-Host "Restarting trigger (deactivate -> activate) ..."
  try {
    $null = Invoke-RestMethod -Method Post -Uri "$apiBase/v1/workflows/$WorkflowId/deactivate" -Headers $headers
    Start-Sleep -Seconds 2
    $null = Invoke-RestMethod -Method Post -Uri "$apiBase/v1/workflows/$WorkflowId/activate" -Headers $headers
    Write-Host "Trigger restarted."
  } catch {
    Write-Warning "Trigger restart failed — activate manually in n8n UI if needed."
    Write-Warning $_
  }
}

# ── ACTIVATE (explicit flag) ───────────────────────────────────────────────────

if ($Activate.IsPresent) {
  Write-Host "Activating workflow $WorkflowId ..."
  try {
    $null = Invoke-RestMethod -Method Post -Uri "$apiBase/v1/workflows/$WorkflowId/activate" -Headers $headers
    Write-Host "Activated."
  } catch {
    Write-Warning "Activation failed — activate manually in n8n UI."
    Write-Warning $_
  }
}

Write-Host ""
Write-Host "Done. Workflow ID: $WorkflowId"
