<#
.SYNOPSIS
  Wrapper for the Content Factory workflow. Calls the universal publish.ps1.
  Keeps N8N_WORKFLOW_ID as the env key for backward compatibility.

.EXAMPLE
  .\scripts\publish-workflow.ps1              # update
  .\scripts\publish-workflow.ps1 -Activate    # update + activate
  .\scripts\publish-workflow.ps1 -Create      # create new
  .\scripts\publish-workflow.ps1 -Status      # check status
  .\scripts\publish-workflow.ps1 -Deactivate  # deactivate
  .\scripts\publish-workflow.ps1 -Execute     # manual test run on VPS
  .\scripts\publish-workflow.ps1 -Review      # inspect live vs local
  .\scripts\publish-workflow.ps1 -Executions  # list last 10 executions
  .\scripts\publish-workflow.ps1 -ListAll     # list all workflows
#>
param(
  [switch]$Create,
  [switch]$Activate,
  [switch]$Deactivate,
  [switch]$Status,
  [switch]$Execute,
  [switch]$Executions,
  [int]$ExecutionsLimit = 10,
  [switch]$Review,
  [switch]$ListAll
)

& "$PSScriptRoot\publish.ps1" `
  -WorkflowFile "workflows/test1-content-factory.workflow.json" `
  -EnvKey "N8N_WORKFLOW_ID" `
  -Create:$Create `
  -Activate:$Activate `
  -Deactivate:$Deactivate `
  -Status:$Status `
  -Execute:$Execute `
  -Executions:$Executions `
  -ExecutionsLimit:$ExecutionsLimit `
  -Review:$Review `
  -ListAll:$ListAll
