# PowerShell helper to download sample Project Aria sequences for Boxer
# Usage:
#   .\scripts\download_aria_data.ps1             # download all sequences
#   .\scripts\download_aria_data.ps1 hohen_gen1  # download one sequence

$ErrorActionPreference = 'Stop'

$DataDir = "sample_data"
$BaseUrl = "https://huggingface.co/datasets/facebook/boxer/resolve/main"

$AllSeqs = @(
    "hohen_gen1"
    "nym10_gen1"
    "cook0_gen2"
)

$Files = @(
    "main.vrs"
    "closed_loop_trajectory.csv"
    "online_calibration.jsonl"
    "semidense_observations.csv.gz"
    "semidense_points.csv.gz"
)

function Download-File {
    param($Url, $Dest)
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
}

function Download-Seq {
    param($Seq)
    $SeqDir = Join-Path $DataDir $Seq
    New-Item -ItemType Directory -Force -Path $SeqDir | Out-Null
    Write-Host "Downloading $Seq ..."
    foreach ($f in $Files) {
        $Dest = Join-Path $SeqDir $f
        if (Test-Path $Dest) {
            Write-Host "  Already exists: $f"
            continue
        }
        Write-Host "  $f"
        Download-File "$BaseUrl/$Seq/$f" $Dest
    }
}

if ($args.Count -gt 0) {
    foreach ($seq in $args) {
        Download-Seq $seq
    }
} else {
    foreach ($seq in $AllSeqs) {
        Download-Seq $seq
    }
}

Write-Host "Done. Data saved to $DataDir/"
