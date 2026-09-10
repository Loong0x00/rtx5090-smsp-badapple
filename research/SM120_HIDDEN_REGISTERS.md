# SM120 hidden PTX/SASS register probe

Test platform:

- GPU: GeForce RTX 5090 / GB202, 170 SMs
- Compute capability: 12.0 (`sm_120`)
- Driver: 610.57.04
- CUDA / ptxas / nvdisasm: 13.3

This is reverse-engineering evidence for this exact stack, not an NVIDIA ABI.

## PTX parser probe

Minimal PTX modules attempted `mov.u32 r, %candidate` and were assembled with
`ptxas -arch=sm_120`.

Accepted:

- `%warpid`
- `%smid`
- `%nsmid`
- `%envreg0` (and the documented envreg family)
- `%pm0` (and the documented PM family)

Rejected:

- `%schedulerid`, `%scheduler_id`, `%affinity`
- `%smspid`, `%smsp_id`
- `%subpartitionid`, `%subpartition_id`
- `%partitionid`, `%processingblockid`
- `%warpschedulerid`, `%warpscheduler_id`
- `%virtid`, `%virtcfg`, `%virtualsmid`

Searching strings in CUDA 13.3 `ptxas` found the under-specified `%envregN` and
`%pmN` families, but no SMSP/scheduler-ID spelling.

## `%envregN` and `%pmN` silicon dump

A cooperative 170-CTA x 4-warp launch, limited to one CTA/SM, read all 32
environment registers and all eight PM registers. Every register had one value
across all 680 sampled warps; none distinguished the four warp/SMSP paths.

Non-zero environment values in this launch:

```text
envreg0  = 04b032e6   envreg1  = 00000100
envreg2  = 04c00000   envreg6  = 00000001
envreg10 = 00000120   envreg26 = 000fffff
envreg29 = 00000100   envreg30 = 02000000
envreg31 = 00000100
```

`pm0..pm7` all returned zero because no private performance event selectors
were configured. PTX defines the readout path but not useful fixed events.

## Raw SASS system-register probe

SM120 `S2R` carries an 8-bit system-register selector in instruction bits
`[79:72]`. A minimal cubin containing `S2R R7, SR_LANEID` was patched at that
selector and loaded in a fresh CUDA context.

Known read-only selector results included:

```text
0x02 SR_VIRTCFG          = 2aa83020 (constant)
0x03 SR_VIRTID           = (physical_warpid << 8) | laneid
0x0f SR_ORDERING_TICKET  = 00000000
0x1c SR_AFFINITY         = 00000000 for every lane/warp
0x2c SR_SM_SPA_VERSION   = a04a0400
0x43 SR_VIRTUALSMID      = SM ID (zero in the one-block sample)
0x44 SR_VIRTUALENGINEID  = 0000003f for all four warps
0x84 SR_VARIABLE_RATE    = 00000000
```

`SR_AFFINITY`, `SR_VIRTUALENGINEID`, and `SR_ORDERING_TICKET` therefore did
**not** expose an SMSP ID in this compute-kernel experiment. `SR_VIRTID` only exposed the already-known lane and
physical warp-slot IDs. For a 128-thread sole-resident CTA it returned:

```text
warp 0: 0x00000000..0x0000001f
warp 1: 0x00000100..0x0000011f
warp 2: 0x00000200..0x0000021f
warp 3: 0x00000300..0x0000031f
```

`nvdisasm`'s complete 256-selector naming sweep (also represented by Cubit's
SM120 table) contains no named scheduler/SMSP-ID register.

## Empirical SMSP formula

`smsp_allocation_probe` now tests two kinds of four-warp groups:

- physical warp slots `0,1,2,3`: run concurrently in about one single-warp time;
- slots `0,4,8,12` (and the other same-low-two-bit groups): take about the
  four-warp contended time.

On this GB202, the measurements support:

```text
SMSP = physical_warpid & 3
```

This is stronger evidence than merely observing four different `%warpid`
values, but remains an implementation-specific measurement.

## Nsight Compute warp-ID sampling

With authorized performance-counter access, CUDA 13.3 successfully collected:

```bash
sudo -E ncu --kernel-name-base demangled \
  -k 'regex:.*ilp_ffma_probe.*' -c 1 \
  --print-metric-instances details \
  --metrics smsp__warpidsamp_warps_issue_stalled_selected \
  ./smsp_allocation_probe 100000 1
```

The metric exposes documented `<SMSP ID>:<Warp ID>` instance keys. One sampled
launch reported keys such as `0:0`, `0:4`, `0:8`, and `0:12`. These identifiers
are local/aggregated sampling categories and omit a physical SM coordinate;
they do not provide 680 globally unique, simultaneous SMSP channels. Profiling
also replays/perturbs execution, so this is useful corroboration rather than a
real-time physical framebuffer.

## Deliberately not attempted

Unknown SASS selectors that `nvdisasm` does not name were not executed. A full
`0x00..0xff` silicon sweep on the only/display GPU can trigger illegal
instructions, context loss or a GPU reset. If pursued, it should use a spare
GPU, one selector per fresh process, a short external timeout and automatic
recovery.

PTX `pmevent` (SASS `PMTRIG`) can emit programmable performance-monitor
triggers, but it does not expose 680 independently identified live SMSP values;
normal profiling rolls such events up and requires external counter setup.

No PTX or SASS instruction was found that writes directly to the display engine
or scanout framebuffer. SM code still needs CUDA/graphics external-memory
interop (or an OS graphics API) to present pixels; the display engine is a
separate hardware block.

## References

- NVIDIA PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- NVIDIA CUDA Binary Utilities: <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- Cubit SM120 assembler/tables: <https://github.com/kacper-daftcode/cubit>
- Blackwell ISA database: <https://github.com/kacper-daftcode/blackwell-isa>
- OpenPTXas SM120 work: <https://github.com/garrick99/openptxas>
