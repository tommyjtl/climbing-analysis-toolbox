MP4_DIR="/Volumes/Climbing Videos/2025/Blue Tag - June 2 - Mosaic/attempts"                  # Directory containing your .mp4 files
REF_IMG="/Volumes/Climbing Videos/2025/Blue Tag - June 2 - Mosaic/attempts/reference.jpg"    # Path to your reference image

for mp4file in "$MP4_DIR"/*.mp4; do
  [ -e "$mp4file" ] || continue  # skip if no .mp4 files
  echo "Processing '$mp4file'..."
  python ../src/cruxes/warp_video.py \
    --src_video_path "$mp4file" \
    --ref_img "$REF_IMG" \
    --type "fixed"
done

echo "Batch processing done!"