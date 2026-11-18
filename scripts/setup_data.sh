#!/usr/bin/env bash
# Downloads everything the pipeline needs (~6.4 GB) into ./data and ./checkpoints.
# nuScenes is released under CC BY-NC-SA 4.0 -- by downloading you accept its terms (nuscenes.org/terms-of-use).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/nuscenes checkpoints
[ -f checkpoints/bevformer_tiny_epoch_24.pth ] || curl -L -o checkpoints/bevformer_tiny_epoch_24.pth \
  https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
for name in bevformer_small_epoch_24.pth bevformer_r101_dcn_24ep.pth; do
  [ -f "checkpoints/$name" ] || curl -L -o "checkpoints/$name" "https://github.com/zhiqi-li/storage/releases/download/v1.0/$name"
done
if [ ! -d data/nuscenes/v1.0-mini ]; then
  curl -L -o data/nuscenes/v1.0-mini.tgz https://www.nuscenes.org/data/v1.0-mini.tgz
  tar -xzf data/nuscenes/v1.0-mini.tgz -C data/nuscenes
fi
if [ ! -d data/nuscenes/can_bus ]; then
  curl -L -o data/nuscenes/can_bus.zip https://motional-nuscenes.s3.amazonaws.com/public/v1.0/can_bus.zip
  unzip -q data/nuscenes/can_bus.zip -d data/nuscenes
fi
echo "Done."
