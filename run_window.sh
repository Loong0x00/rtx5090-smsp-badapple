#!/usr/bin/env bash
set -euo pipefail

if (($# > 5)); then
    echo "Usage: $0 [VIDEO=media/badapple-official.mp4] [FPS=20] [BURN_MS=8] [DURATION] [SCALE=32]" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
video=${1:-"$root/media/badapple-official.mp4"}
fps=${2:-20}
burn_ms=${3:-8}
duration=${4:-}
scale=${5:-32}

validate_range() {
    local name=$1 value=$2 minimum=$3 maximum=$4
    if ! [[ "$value" =~ ^([0-9]+([.][0-9]*)?|\.[0-9]+)$ ]] ||
       ! LC_ALL=C awk -v x="$value" -v lo="$minimum" -v hi="$maximum" \
           'BEGIN { exit !(x >= lo && x <= hi) }'; then
        echo "$name must be a finite decimal in [$minimum, $maximum], got '$value'" >&2
        exit 2
    fi
}

validate_range FPS "$fps" 0.1 240
validate_range BURN_MS "$burn_ms" 0 100
if [[ -n "$duration" ]]; then
    validate_range DURATION "$duration" 0.001 86400
fi
if [[ ! -f "$video" ]]; then
    echo "Video not found: $video" >&2
    exit 1
fi
if ! [[ "$scale" =~ ^[1-9][0-9]*$ ]] || ((${#scale} > 3)) ||
   ((10#$scale > 128)); then
    echo "SCALE must be an integer in [1, 128]" >&2
    exit 2
fi
if [[ ! -x "$root/smsp_badapple" ]]; then
    make -C "$root"
fi

width=$((34 * scale))
height=$((20 * scale))
duration_args=()
if [[ -n "$duration" ]]; then
    duration_args=(-t "$duration")
fi

# Input video -> 34x20 binary frames -> 680 SMSP/warp kernel -> raw GPU-returned
# frames -> nearest-neighbour window. Press q or Escape in ffplay to quit.
ffmpeg -hide_banner -loglevel warning -re -i "$video" "${duration_args[@]}" \
    -vf "fps=${fps},scale=34:20:force_original_aspect_ratio=decrease,pad=34:20:(ow-iw)/2:(oh-ih)/2:black,format=gray,lut=y='if(gte(val,128),255,0)'" \
    -an -f rawvideo -pix_fmt gray - \
  | "$root/smsp_badapple" --raw - --raw-output --fps "$fps" --burn-ms "$burn_ms" \
  | ffplay -hide_banner -loglevel warning -autoexit \
      -f rawvideo -pixel_format gray -video_size 34x20 -framerate "$fps" \
      -vf "scale=${width}:${height}:flags=neighbor" \
      -window_title "RTX 5090 — 680 SMSP Bad Apple" -
