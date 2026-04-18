# PowerShell script to download BoxerNet, DinoV3, and OWLv2 model checkpoints
# Usage: .\scripts\download_ckpts.ps1

$ErrorActionPreference = 'Stop'

$CkptDir = "ckpts"
$BaseUrl = "https://huggingface.co/facebook/boxer/resolve/main"

$Files = @(
    "boxernet_hw960in4x6d768-wssxpf9p.ckpt",
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "owlv2-base-patch16-ensemble.pt"
)

# Create checkpoint directory
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null

foreach ($f in $Files) {
    $FilePath = Join-Path $CkptDir $f
    if (Test-Path $FilePath) {
        Write-Host "Already exists: $FilePath"
        continue
    }
    Write-Host "Downloading $f ..."
    $Url = "$BaseUrl/$f"
    $MaxRetries = 3
    $Retry = 0
    while ($Retry -lt $MaxRetries) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $FilePath -UseBasicParsing
            Write-Host "Downloaded: $f"
            break
        } catch {
            $Retry++
            if ($Retry -lt $MaxRetries) {
                Write-Host "Download failed, retrying ($Retry/$MaxRetries)..."
                Start-Sleep -Seconds 5
            } else {
                Write-Host "Failed to download $f after $MaxRetries attempts"
                throw $_
            }
        }
    }
}

Write-Host "Done. Checkpoints saved to $CkptDir/"
