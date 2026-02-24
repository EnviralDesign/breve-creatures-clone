param(
    [string]$BaseUrl = "http://127.0.0.1:8787",
    [int]$Generations = 120,
    [int]$PollSeconds = 1,
    [string]$OutCsv = "data/benchmarks/neural-v2-benchmark.csv",
    [int]$PopulationSize = 0
)

$ErrorActionPreference = "Stop"

function Get-Json([string]$Url) {
    return Invoke-RestMethod -Uri $Url -Method Get
}

function Post-Json([string]$Url, [object]$Body) {
    $payload = ($Body | ConvertTo-Json -Depth 8)
    return Invoke-RestMethod -Uri $Url -Method Post -ContentType "application/json" -Body $payload
}

if (-not (Test-Path -Path (Split-Path -Path $OutCsv -Parent))) {
    New-Item -ItemType Directory -Path (Split-Path -Path $OutCsv -Parent) -Force | Out-Null
}

$stateUrl = "$BaseUrl/api/evolution/state"
$summaryUrl = "$BaseUrl/api/evolution/performance/summary"
$controlUrl = "$BaseUrl/api/evolution/control"

if ($PopulationSize -gt 0) {
    Write-Host "Applying benchmark population size $PopulationSize and restarting evolution"
    Post-Json $controlUrl @{
        action = "set_population_size"
        populationSize = $PopulationSize
    } | Out-Null
    Post-Json $controlUrl @{
        action = "restart"
    } | Out-Null
    $deadline = (Get-Date).AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 250
        $state = Get-Json $stateUrl
        if ([int]$state.generation -eq 1 -and [int]$state.currentAttemptIndex -eq 0 -and [int]$state.currentTrialIndex -eq 0) {
            break
        }
    } while ((Get-Date) -lt $deadline)
}

$initialState = Get-Json $stateUrl
$startGeneration = [int]$initialState.generation
$targetGeneration = $startGeneration + $Generations

Write-Host "Starting benchmark at generation $startGeneration; target generation $targetGeneration"
Post-Json $controlUrl @{
    action = "queue_fast_forward"
    fastForwardGenerations = $Generations
} | Out-Null

$rows = New-Object System.Collections.Generic.List[object]
$seenGenerations = New-Object System.Collections.Generic.HashSet[int]

while ($true) {
    $summary = Get-Json $summaryUrl
    $generation = [int]$summary.generation
    if (-not $seenGenerations.Contains($generation)) {
        $seenGenerations.Add($generation) | Out-Null
        $rows.Add([PSCustomObject]@{
            generation = $generation
            bestEverFitness = [double]$summary.bestEverFitness
            recentBestFitness = [double]$summary.recentBestFitness
            stagnationGenerations = [int]$summary.stagnationGenerations
            diversityState = [string]$summary.diversityState
            mutationRate = [double]$summary.mutationPressure.currentRate
            atLowerClamp = [bool]$summary.mutationPressure.atLowerClamp
            atUpperClamp = [bool]$summary.mutationPressure.atUpperClamp
            timestamp = (Get-Date).ToString("o")
        }) | Out-Null
        Write-Host ("gen={0} bestEver={1:N3} recent={2:N3} stagnation={3}" -f `
            $generation, $summary.bestEverFitness, $summary.recentBestFitness, $summary.stagnationGenerations)
    }

    if ($generation -ge $targetGeneration) {
        break
    }

    Start-Sleep -Seconds $PollSeconds
}

$rows | Sort-Object generation | Export-Csv -Path $OutCsv -NoTypeInformation -Force
Write-Host "Benchmark complete. Wrote $($rows.Count) rows to $OutCsv"
