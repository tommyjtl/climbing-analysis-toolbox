#!/bin/bash

DIR="/Users/tommyjtl/Documents/Projects/climbing/climbs/apr-15"
REF_IMG="$DIR/warped.png"

shopt -s nullglob

for video in "$DIR"/*.MOV; do
  echo "Processing: $(basename "$video")"

  python body_trajectory_demo.py \
    --pose_backend "mediapipe" --smoothing "gaussian" \
    --video_path "$video"

  echo "Done: $(basename "$video")"
  echo
done

