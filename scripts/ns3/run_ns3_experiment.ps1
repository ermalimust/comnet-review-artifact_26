param(
    [Parameter(Mandatory = $true)]
    [string]$Ns3Root,

    [int[]]$Seeds = @(7, 11, 13, 17, 19),

    [string[]]$Scenarios = @("overall", "load_high", "vacation_high", "drift_strong"),

    [ValidateSet("video", "event_c2")]
    [string]$Target = "video",

    [double]$Duration = 120.0,

    [double]$Warmup = 5.0,

    [double]$Window = 10.0,

    [double]$ViolationTau = 0.01,

    [string]$OutDir = "paper1_draft/experiment_outputs/ns3_validation"
)

$ErrorActionPreference = "Stop"

function Convert-ToMsysPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolved = (Resolve-Path $Path).Path
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $Matches[1].ToLowerInvariant()
        $tail = $Matches[2] -replace "\\", "/"
        return "/$drive/$tail"
    }
    return ($resolved -replace "\\", "/")
}

function Invoke-Ns3Program {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProgramArgs,

        [Parameter(Mandatory = $true)]
        [string]$Ns3RootResolved,

        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $localMsysBash = Join-Path $RepoRoot "tools\ns3\msys64\usr\bin\bash.exe"
    $ns3Script = Join-Path $Ns3RootResolved "ns3"
    $wafScript = Join-Path $Ns3RootResolved "waf"

    if (Test-Path $localMsysBash) {
        $ns3RootMsys = Convert-ToMsysPath $Ns3RootResolved
        $escapedProgramArgs = $ProgramArgs -replace "'", "'\\''"
        $bashCommand = "export HOME=/tmp; export PATH=/mingw64/bin:/usr/bin:`$PATH; cd `"$ns3RootMsys`" && ./ns3 run '$escapedProgramArgs'"
        & $localMsysBash -lc $bashCommand
    }
    elseif (Test-Path $ns3Script) {
        Push-Location $Ns3RootResolved
        try {
            python $ns3Script run $ProgramArgs
        }
        finally {
            Pop-Location
        }
    }
    elseif (Test-Path $wafScript) {
        Push-Location $Ns3RootResolved
        try {
            python $wafScript --run $ProgramArgs
        }
        finally {
            Pop-Location
        }
    }
    else {
        throw "Could not find ns3 or waf in $Ns3RootResolved"
    }

    if ($LASTEXITCODE -ne 0) {
        throw "ns-3 run failed with exit code ${LASTEXITCODE}: $ProgramArgs"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ns3RootResolved = (Resolve-Path $Ns3Root).Path
$scratchSource = Join-Path $repoRoot "experiments\ns3\uav_vehicular_vacation.cc"
$scratchDir = Join-Path $ns3RootResolved "scratch"
$scratchTarget = Join-Path $scratchDir "uav_vehicular_vacation.cc"
$outDirResolved = Join-Path $repoRoot $OutDir
$packetDir = Join-Path $outDirResolved "packets"

if (!(Test-Path $scratchSource)) {
    throw "Missing scratch source: $scratchSource"
}
if (!(Test-Path $scratchDir)) {
    throw "Missing ns-3 scratch directory: $scratchDir"
}

New-Item -ItemType Directory -Force -Path $packetDir | Out-Null
Copy-Item -LiteralPath $scratchSource -Destination $scratchTarget -Force

foreach ($scenario in $Scenarios) {
    foreach ($seed in $Seeds) {
        $packetPath = Join-Path $packetDir ("{0}_seed{1}_packets.csv" -f $scenario, $seed)
        $packetPathArg = $packetPath -replace "\\", "/"
        $programArgs = "uav_vehicular_vacation --scenario=$scenario --target=$Target --seed=$seed --duration=$Duration --warmup=$Warmup --out=$packetPathArg"
        Write-Host "Running ns-3: $programArgs"
        Invoke-Ns3Program -ProgramArgs $programArgs -Ns3RootResolved $ns3RootResolved -RepoRoot $repoRoot
    }
}

$postprocess = Join-Path $repoRoot "experiments\ns3\postprocess_ns3.py"
$packetGlob = Join-Path $packetDir "*_packets.csv"
python $postprocess --input $packetGlob --outdir $outDirResolved --window $Window --warmup $Warmup --violation-tau $ViolationTau

Write-Host "NS-3 validation outputs written to $outDirResolved"
