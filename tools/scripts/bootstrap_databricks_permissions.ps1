param(
    [Parameter(Mandatory = $true)][string]$Principal,
    [Parameter(Mandatory = $true)][string]$ModelName,
    [string]$FeatureTable = "workspace.default.iris_features",
    [string]$Catalog = "workspace",
    [string]$Schema = "default"
)

$ErrorActionPreference = "Stop"

function Set-Grant([string]$Type, [string]$Name, [string[]]$Privileges) {
    $body = @{ changes = @(@{ principal = $Principal; add = $Privileges }) } | ConvertTo-Json -Depth 5 -Compress
    databricks grants update $Type $Name --json $body
    if ($LASTEXITCODE -ne 0) { throw "No fue posible aplicar grants sobre $Type $Name" }
}

databricks current-user me
if ($LASTEXITCODE -ne 0) { throw "La cuenta Databricks no está disponible o la sesión expiró." }

Set-Grant "CATALOG" $Catalog @("USE_CATALOG")
Set-Grant "SCHEMA" "$Catalog.$Schema" @("USE_SCHEMA")
Set-Grant "TABLE" $FeatureTable @("SELECT")
Set-Grant "REGISTERED_MODEL" $ModelName @("APPLY_TAG", "EXECUTE", "MANAGE")

Write-Host "Permisos de Unity Catalog aplicados. Los permisos CAN_MANAGE_RUN del job y CAN_MANAGE del endpoint se asignan después del primer bundle deploy."
