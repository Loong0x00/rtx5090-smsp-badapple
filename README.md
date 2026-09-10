# RTX 5090 SMSP Bad Apple experiment

The RTX 5090 has 170 enabled SMs and four SM subpartitions (SMSPs) per SM:

```text
170 x 4 = 680 = 34 x 20 pixels
```

Tested on an RTX 5090 with compute capability 12.0, driver 610.57.04 and CUDA
13.3. Building requires `nvcc`; video playback additionally uses `ffmpeg` and
`ffplay`.

This experiment launches 170 cooperative CTAs, limits residency to one CTA per
SM with dynamic shared memory, and puts four warps in every CTA. On the tested
GB202, the four resident warps empirically occupy four independent SMSP issue
paths. Each logical warp owns one pixel of a 34x20 binary frame for that launch.

Inline PTX reads `%smid` and `%warpid`. There is no documented `%smspid`; the
program therefore verifies the observable invariants (170 unique SMs and four
distinct physical warp slots per CTA) without claiming a documented warp-ID to
SMSP-number formula. `smsp_allocation_probe` additionally measures every pair
of the first four warps. On the tested GB202 they execute concurrently without
sharing an issue path, empirically confirming that they occupy four distinct
scheduler/SMSP paths.

A white pixel runs a timed dependent FP32 FMA loop. A black pixel mostly executes
`nanosleep`. Timing uses PTX `%globaltimer` nanoseconds rather than peak-clock
estimates, so DVFS does not stretch the requested interval in cycle units. With
activity enabled, white output is emitted only after the FMA loop executes and
changes its accumulator; per-pixel activity records are verified on the host.
The returned 34x20 logical framebuffer is enlarged in the terminal.

## Build and probe

```bash
make
./smsp_badapple --probe
./smsp_allocation_probe
```

The allocation microbenchmark compares one-warp, every two-warp pair,
four-warp, sixteen-warp, and stride-four warp-slot groups. On the tested GB202,
slots 0/1/2/3 do not contend while 0/4/8/12 do, supporting the empirical formula
`SMSP = physical_warpid & 3`. It is not a supported PTX affinity API.

Results from probing hidden PTX names, `%envregN`, `%pmN`, and patched SM120
SASS system-register selectors are recorded in
[`research/SM120_HIDDEN_REGISTERS.md`](research/SM120_HIDDEN_REGISTERS.md).

Inspect compiler-generated PTX:

```bash
make ptx
grep -nE '%(smid|warpid)' smsp_badapple.ptx
```

## Built-in test animation

```bash
./smsp_badapple --demo --fps 20 --burn-ms 8 --frames 120
```

`--burn-ms 0` disables the activity-encoding loop and deliberately becomes a
mapping-only/pass-through mode. Accepted limits are 0.1–240 FPS, either zero
or 0.000001–100 ms burn, and 1–1,000,000 demo frames. Values must be finite and
fully parsed, and exactly one of `--probe`, `--demo` or `--raw` is required. A
warning is printed if burn time exceeds the requested frame period.

## Play Bad Apple

Media is not redistributed in this repository. Supply your own local video, or
retrieve the public Alstroemeria Records upload referenced in
[`media/SOURCE.txt`](media/SOURCE.txt):

```bash
mkdir -p media
yt-dlp --no-playlist -f 'bv*[height<=720]/b[height<=720]' \
  --remux-video mp4 -o 'media/badapple-official.%(ext)s' \
  'https://www.youtube.com/watch?v=i41KoE0iMYU'
./run_video.sh
```

For a real pixel window instead of terminal blocks, pipe the GPU-returned raw
frames to ffplay with nearest-neighbour enlargement:

```bash
./run_window.sh
```

The displayed window is 1088x640 by default (each SMSP pixel becomes 32x32).
Press `q` or Escape to close it. The optional fifth argument changes the pixel
scale.

A different local file, frame rate and per-frame load duration can be supplied:

```bash
./run_video.sh /path/to/video.mp4 20 8
```

The optional fourth argument limits playback duration in seconds, which is
useful for a quick test:

```bash
./run_video.sh media/badapple-official.mp4 20 2 1
```

The wrapper uses ffmpeg to scale, letterbox, grayscale and threshold the video
to raw 34x20 frames. Audio is intentionally omitted; the downloaded format is
video-only.

## Caveat

NVIDIA PTX does not expose an SMSP ID or promise the warp-to-SMSP allocation
formula. Four warps in a sole resident CTA are the architecture-specific method
being tested here. The built-in probe proves only simultaneous 170-SM coverage,
one-SM-per-CTA and four distinct physical warp slots; SMSP independence relies
on `smsp_allocation_probe` contention measurements.

`%smid`/`%warpid` are diagnostic, volatile special registers, and block-index
order is not physical die order. Every frame launch rechecks sampled coverage,
verifies that `%smid/%warpid` match at the beginning and end of each warp's
activity interval, and reports whether logical slots moved relative to the
initial launch. Therefore the
output is a logical 34x20 activity encoding, not a stable physical-die
framebuffer. A persistent kernel or an experimentally calibrated physical-ID
mapping would be required for stronger placement stability.
