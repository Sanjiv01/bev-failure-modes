# Downloads everything the pipeline needs (~5.4 GB) into ./data and ./checkpoints.
# nuScenes is released under CC BY-NC-SA 4.0 -- by downloading you accept its terms (nuscenes.org/terms-of-use).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force "$root\data\nuscenes", "$root\checkpoints" | Out-Null

$ckpt = "$root\checkpoints\bevformer_tiny_epoch_24.pth"
if (-not (Test-Path $ckpt)) {
  Write-Host "BEVFormer-tiny checkpoint (400 MB, official GitHub release)"
  curl.exe -L -o $ckpt https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
}
if (-not (Test-Path "$root\data\nuscenes\v1.0-mini")) {
  Write-Host "nuScenes v1.0-mini (4.2 GB)"
  curl.exe -L -o "$root\data\nuscenes\v1.0-mini.tgz" https://www.nuscenes.org/data/v1.0-mini.tgz
  tar -xzf "$root\data\nuscenes\v1.0-mini.tgz" -C "$root\data\nuscenes"
}
if (-not (Test-Path "$root\data\nuscenes\can_bus")) {
  Write-Host "nuScenes CAN bus expansion (780 MB) - BEVFormer uses it for ego motion"
  curl.exe -L -o "$root\data\nuscenes\can_bus.zip" https://motional-nuscenes.s3.amazonaws.com/public/v1.0/can_bus.zip
  Expand-Archive "$root\data\nuscenes\can_bus.zip" -DestinationPath "$root\data\nuscenes"
}
Write-Host "Done."
