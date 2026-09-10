#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(_err));                    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr double kMinFps = 0.1;
constexpr double kMaxFps = 240.0;
constexpr double kMinNonzeroBurnMs = 0.000001;  // one nanosecond
constexpr double kMaxBurnMs = 100.0;
constexpr int kMaxDemoFrames = 1'000'000;
constexpr uint64_t kProbeBurnNs = 100'000;

struct WarpPlacement {
    uint32_t smid;
    uint32_t warpid;
    uint32_t end_smid;
    uint32_t end_warpid;
};

__device__ __forceinline__ uint32_t ptx_smid() {
    uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ uint32_t ptx_warpid() {
    uint32_t value;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ uint64_t ptx_globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Exactly four warps are used by each CTA, and dynamic shared memory limits
// residency to one CTA/SM. On this tested GB202, the separate allocation probe
// shows that those four warps occupy four non-contending SMSP issue paths. PTX
// exposes SM/warp-slot IDs, but neither an smspid nor a placement guarantee.
__global__ void smsp_frame_kernel(const uint8_t* input,
                                  uint8_t* output,
                                  float* sink,
                                  uint32_t* activity,
                                  WarpPlacement* placement,
                                  uint64_t burn_ns) {
    extern __shared__ uint32_t occupancy_limiter[];

    const uint32_t lane = threadIdx.x & (kWarpSize - 1);
    const uint32_t logical_warp = threadIdx.x / kWarpSize;
    const uint32_t pixel = blockIdx.x * kWarpsPerBlock + logical_warp;

    // Make the dynamic shared-memory allocation observable to the compiler.
    if (threadIdx.x == 0) {
        occupancy_limiter[0] = blockIdx.x;
    }
    __syncthreads();

    const uint32_t sm = ptx_smid();
    const uint32_t physical_warp = ptx_warpid();
    const uint8_t value = input[pixel];
    const bool white = value >= 128;
    const float initial_x =
        1.0f + static_cast<float>((pixel + lane) & 7u) * 0.001f;
    float x = initial_x;
    uint32_t loops = 0;

    // Encode a white pixel as FP32 issue activity and a black pixel as mostly
    // sleeping. %globaltimer is a wall-clock-like nanosecond counter, so this
    // interval does not scale with SM DVFS as a clock64() interval would.
    if (burn_ns != 0) {
        const uint64_t begin = ptx_globaltimer_ns();
        if (white) {
            do {
#pragma unroll 32
                for (int i = 0; i < 32; ++i) {
                    asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                                 : "+f"(x)
                                 : "f"(1.00000011920928955078125f),
                                   "f"(0.00000011920928955078125f));
                }
                ++loops;
            } while (ptx_globaltimer_ns() - begin < burn_ns);
        } else {
            do {
                __nanosleep(1000);
                ++loops;
            } while (ptx_globaltimer_ns() - begin < burn_ns);
        }
    }

    if (lane == 0) {
        placement[pixel] = {sm, physical_warp, ptx_smid(), ptx_warpid()};
        sink[pixel] = x;
        activity[pixel] = (white ? 0x80000000u : 0u) |
                          (loops & 0x7fffffffu);

        // With activity enabled, a white output now depends on the FMA path
        // actually executing and changing its accumulator. A zero-duration
        // run remains a deliberate mapping-only/pass-through mode.
        if (burn_ns == 0) {
            output[pixel] = value;
        } else if (white) {
            output[pixel] = (loops != 0 && x != initial_x) ? 255 : 0;
        } else {
            output[pixel] = loops != 0 ? 0 : 255;
        }
    }
}

struct Options {
    std::string raw_path;
    double fps = 20.0;
    double burn_ms = 8.0;
    int demo_frames = 120;
    bool no_ansi = false;
    bool raw_output = false;
    bool probe_only = false;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --probe\n"
        "  %s --demo [--fps N] [--burn-ms N] [--frames N]\n"
        "  %s --raw frames.gray [--fps N] [--burn-ms N] [--raw-output]\n\n"
        "Choose exactly one mode: --probe, --demo, or --raw.\n"
        "Limits: fps %.1f..%.0f, burn-ms 0 or %.6g..%.0f, frames 1..%d.\n"
        "Raw input/output is tightly packed 34x20 8-bit grayscale on an RTX 5090.\n"
        "Use '-' as the raw path to read ffmpeg output from stdin.\n"
        "--raw-output writes only GPU-returned frames to stdout (for ffplay).\n",
        argv0, argv0, argv0, kMinFps, kMaxFps, kMinNonzeroBurnMs,
        kMaxBurnMs, kMaxDemoFrames);
}

static double parse_finite_double(const char* flag,
                                  const char* text,
                                  double minimum,
                                  double maximum) {
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (text == end || end == nullptr || *end != '\0' || errno == ERANGE ||
        !std::isfinite(value) || value < minimum || value > maximum) {
        std::fprintf(stderr, "%s must be a finite number in [%.3g, %.3g], got '%s'\n",
                     flag, minimum, maximum, text);
        std::exit(2);
    }
    return value;
}

static int parse_bounded_int(const char* flag,
                             const char* text,
                             int minimum,
                             int maximum) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text == end || end == nullptr || *end != '\0' || errno == ERANGE ||
        value < minimum || value > maximum) {
        std::fprintf(stderr, "%s must be an integer in [%d, %d], got '%s'\n",
                     flag, minimum, maximum, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

static Options parse_options(int argc, char** argv) {
    Options o;
    int mode_count = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "%s requires a value\n", flag);
                std::exit(2);
            }
            return argv[i];
        };
        if (arg == "--probe") {
            o.probe_only = true;
            ++mode_count;
        } else if (arg == "--demo") {
            ++mode_count;
        } else if (arg == "--raw") {
            o.raw_path = need_value("--raw");
            if (o.raw_path.empty()) {
                std::fprintf(stderr, "--raw path must not be empty\n");
                std::exit(2);
            }
            ++mode_count;
        } else if (arg == "--fps") {
            o.fps = parse_finite_double("--fps", need_value("--fps"),
                                        kMinFps, kMaxFps);
        } else if (arg == "--burn-ms") {
            o.burn_ms = parse_finite_double("--burn-ms", need_value("--burn-ms"),
                                            0.0, kMaxBurnMs);
        } else if (arg == "--frames") {
            o.demo_frames = parse_bounded_int("--frames", need_value("--frames"),
                                              1, kMaxDemoFrames);
        } else if (arg == "--no-ansi") {
            o.no_ansi = true;
        } else if (arg == "--raw-output") {
            o.raw_output = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (mode_count != 1) {
        std::fprintf(stderr,
                     "Exactly one mode must be selected: --probe, --demo, or --raw.\n");
        usage(argv[0]);
        std::exit(2);
    }
    if (o.burn_ms > 0.0 && o.burn_ms < kMinNonzeroBurnMs) {
        std::fprintf(stderr,
                     "--burn-ms must be 0 or at least %.6g ms, got %.17g\n",
                     kMinNonzeroBurnMs, o.burn_ms);
        std::exit(2);
    }
    if (o.burn_ms > 1000.0 / o.fps) {
        std::fprintf(stderr,
                     "Warning: --burn-ms %.3f exceeds the %.3f ms frame period; "
                     "playback cannot sustain %.3f fps.\n",
                     o.burn_ms, 1000.0 / o.fps, o.fps);
    }
    return o;
}

static void make_demo_frame(std::vector<uint8_t>& frame,
                            int width,
                            int height,
                            int frame_number) {
    const double t = frame_number * 0.105;
    const double cx = width * (0.5 + 0.27 * std::sin(t * 0.73));
    const double cy = height * (0.5 + 0.22 * std::cos(t * 0.91));
    const double radius = 3.0 + 2.0 * (0.5 + 0.5 * std::sin(t * 0.57));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double dx = x + 0.5 - cx;
            const double dy = y + 0.5 - cy;
            const bool circle = dx * dx + dy * dy < radius * radius;
            const bool wave = y > height / 2 + 2.5 * std::sin(x * 0.42 + t);
            frame[y * width + x] = (circle ^ wave) ? 255 : 0;
        }
    }
}

static bool draw_terminal(const std::vector<uint8_t>& frame,
                          int width,
                          int height,
                          bool first,
                          bool no_ansi) {
    if (!no_ansi) {
        std::fputs(first ? "\x1b[2J\x1b[H" : "\x1b[H", stdout);
    }
    std::fputs("RTX 5090: 170 SM x 4 logical warps = 680 pixels "
               "(four SMSP paths empirically)\n",
               stdout);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::fputs(frame[y * width + x] >= 128 ? "██" : "  ", stdout);
        }
        std::fputc('\n', stdout);
    }
    return std::fflush(stdout) == 0 && !std::ferror(stdout);
}

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int cooperative = 0;
    int max_optin_shared = 0;
    int shared_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin_shared,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device));
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_per_sm,
                                      cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                      device));

    std::fprintf(stderr,
                 "GPU: %s, CC %d.%d, SMs=%d, cooperative=%s\n"
                 "Shared memory: per-SM=%d, max opt-in/block=%d bytes\n",
                 prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                 cooperative ? "yes" : "no", shared_per_sm, max_optin_shared);

    if (!cooperative) {
        std::fprintf(stderr, "This GPU/runtime does not support cooperative launch.\n");
        return 1;
    }
    if (prop.multiProcessorCount != 170) {
        std::fprintf(stderr,
                     "This experiment expects the RTX 5090's 170 enabled SMs; got %d.\n",
                     prop.multiProcessorCount);
        return 1;
    }

    const int width = 34;
    const int height = 20;
    const size_t pixels = static_cast<size_t>(prop.multiProcessorCount) *
                          kWarpsPerBlock;
    if (pixels != static_cast<size_t>(width * height)) {
        std::fprintf(stderr, "Internal geometry mismatch.\n");
        return 1;
    }

    // Find a dynamic shared-memory amount that allows exactly one block/SM.
    size_t dynamic_shared = static_cast<size_t>(max_optin_shared);
    int active_blocks = 0;
    while (dynamic_shared > 0) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            smsp_frame_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dynamic_shared));
        if (attr_err == cudaSuccess) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, smsp_frame_kernel, kThreadsPerBlock,
                dynamic_shared));
            if (active_blocks == 1) {
                break;
            }
        } else {
            (void)cudaGetLastError();
        }
        dynamic_shared -= std::min<size_t>(dynamic_shared, 1024);
    }
    if (active_blocks != 1) {
        std::fprintf(stderr, "Could not force exactly one resident CTA per SM.\n");
        return 1;
    }
    std::fprintf(stderr,
                 "Launch geometry: %d cooperative CTAs x 4 warps, %zu-byte shared "
                 "occupancy limiter\n",
                 prop.multiProcessorCount, dynamic_shared);

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    float* d_sink = nullptr;
    uint32_t* d_activity = nullptr;
    WarpPlacement* d_placement = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, pixels));
    CUDA_CHECK(cudaMalloc(&d_output, pixels));
    CUDA_CHECK(cudaMalloc(&d_sink, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_activity, pixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_placement, pixels * sizeof(WarpPlacement)));

    auto cleanup = [&]() {
        CUDA_CHECK(cudaFree(d_placement));
        CUDA_CHECK(cudaFree(d_activity));
        CUDA_CHECK(cudaFree(d_sink));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_input));
    };

    std::vector<uint8_t> input(pixels, 0);
    std::vector<uint8_t> output(pixels, 0);
    std::vector<uint32_t> activity(pixels, 0);
    std::vector<WarpPlacement> placement(pixels);
    // The short initial probe deliberately exercises both FMA and sleep paths.
    for (size_t i = 0; i < pixels; ++i) {
        input[i] = ((i / width + i % width) & 1) ? 255 : 0;
    }

    auto launch_frame = [&](uint64_t burn_ns, bool verify_activity) -> bool {
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), pixels,
                              cudaMemcpyHostToDevice));
        void* args[] = {&d_input, &d_output, &d_sink, &d_activity,
                        &d_placement, &burn_ns};
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(smsp_frame_kernel),
            dim3(prop.multiProcessorCount), dim3(kThreadsPerBlock), args,
            dynamic_shared, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, pixels,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(activity.data(), d_activity,
                              pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(placement.data(), d_placement,
                              pixels * sizeof(WarpPlacement),
                              cudaMemcpyDeviceToHost));

        if (!verify_activity || burn_ns == 0) {
            return true;
        }
        bool valid = true;
        size_t failures = 0;
        for (size_t i = 0; i < pixels; ++i) {
            const bool expected_white = input[i] >= 128;
            const bool recorded_white = (activity[i] & 0x80000000u) != 0;
            const uint32_t loops = activity[i] & 0x7fffffffu;
            const uint8_t expected_output = expected_white ? 255 : 0;
            if (recorded_white != expected_white || loops == 0 ||
                output[i] != expected_output) {
                valid = false;
                if (failures++ < 4) {
                    std::fprintf(stderr,
                                 "Activity verification failed at pixel %zu: "
                                 "input=%u marker=%u loops=%u output=%u\n",
                                 i, static_cast<unsigned>(input[i]),
                                 recorded_white ? 1u : 0u, loops,
                                 static_cast<unsigned>(output[i]));
                }
            }
        }
        return valid;
    };

    auto placement_invariant = [&](bool verbose) -> bool {
        bool block_consistent = true;
        bool warp_slots_distinct = true;
        bool samples_stable = true;
        std::unordered_set<uint32_t> unique_sms;
        for (int block = 0; block < prop.multiProcessorCount; ++block) {
            const uint32_t sm = placement[block * 4].smid;
            unique_sms.insert(sm);
            std::unordered_set<uint32_t> warp_slots;
            for (int warp = 0; warp < 4; ++warp) {
                const WarpPlacement p = placement[block * 4 + warp];
                block_consistent &= (p.smid == sm);
                samples_stable &=
                    (p.end_smid == p.smid && p.end_warpid == p.warpid);
                warp_slots.insert(p.warpid);
            }
            warp_slots_distinct &= (warp_slots.size() == 4);
        }

        if (verbose) {
            std::fprintf(stderr,
                         "Warp-slot probe: unique physical SM IDs=%zu/%d, "
                         "one-SM-per-CTA=%s, four distinct warp slots/CTA=%s, "
                         "start/end samples stable=%s\n",
                         unique_sms.size(), prop.multiProcessorCount,
                         block_consistent ? "yes" : "NO",
                         warp_slots_distinct ? "yes" : "NO",
                         samples_stable ? "yes" : "NO");
            std::fprintf(stderr,
                         "First 12 CTA placements (CTA: SM | physical warp slots):\n");
            for (int block = 0;
                 block < std::min(12, prop.multiProcessorCount); ++block) {
                std::fprintf(stderr, "  %3d: %3u | %2u %2u %2u %2u\n", block,
                             placement[block * 4].smid,
                             placement[block * 4 + 0].warpid,
                             placement[block * 4 + 1].warpid,
                             placement[block * 4 + 2].warpid,
                             placement[block * 4 + 3].warpid);
            }
        }
        return unique_sms.size() ==
                   static_cast<size_t>(prop.multiProcessorCount) &&
               block_consistent && warp_slots_distinct && samples_stable;
    };

    // The built-in check proves simultaneous SM coverage and distinct physical
    // warp slots, not SMSP identity. The separate contention microbenchmark
    // supplies the architecture-specific evidence for four SMSP paths.
    if (!launch_frame(kProbeBurnNs, true) || !placement_invariant(true)) {
        std::fprintf(stderr, "Warp-slot or activity-path probe failed.\n");
        cleanup();
        return 1;
    }
    std::fprintf(stderr,
                 "Warp-slot invariant passed; this alone does not prove SMSP "
                 "identity. Run ./smsp_allocation_probe for empirical evidence.\n");
    const std::vector<WarpPlacement> initial_placement = placement;

    if (options.probe_only) {
        cleanup();
        return 0;
    }

    const long double requested_ns =
        static_cast<long double>(options.burn_ms) * 1'000'000.0L;
    const long double max_ns =
        static_cast<long double>(std::numeric_limits<uint64_t>::max());
    if (!std::isfinite(requested_ns) || requested_ns < 0.0L ||
        requested_ns > max_ns - 0.5L) {
        std::fprintf(stderr, "Burn duration cannot be represented in nanoseconds.\n");
        cleanup();
        return 2;
    }
    const uint64_t burn_ns = static_cast<uint64_t>(requested_ns + 0.5L);
    std::fprintf(stderr,
                 "Playback: %.2f fps, %.2f ms globaltimer-based activity/frame\n",
                 options.fps, options.burn_ms);

    FILE* raw = nullptr;
    const bool raw_mode = !options.raw_path.empty();
    if (raw_mode) {
        raw = options.raw_path == "-"
                  ? stdin
                  : std::fopen(options.raw_path.c_str(), "rb");
        if (!raw) {
            std::perror(options.raw_path.c_str());
            cleanup();
            return 1;
        }
    }

    const auto frame_period = std::chrono::duration<double>(1.0 / options.fps);
    auto deadline = std::chrono::steady_clock::now();
    bool first = true;
    bool mapping_warning_printed = false;
    size_t remapped_frames = 0;
    uint64_t frame_number = 0;
    int exit_status = 0;
    while (true) {
        if (raw_mode) {
            const size_t got = std::fread(input.data(), 1, pixels, raw);
            if (got == 0) {
                if (std::ferror(raw)) {
                    std::perror("Failed to read raw input");
                    exit_status = 1;
                }
                break;
            }
            if (got != pixels) {
                if (std::ferror(raw)) {
                    std::perror("Failed to read complete raw frame");
                } else {
                    std::fprintf(stderr, "Truncated raw frame (%zu/%zu bytes).\n",
                                 got, pixels);
                }
                exit_status = 1;
                break;
            }
            for (uint8_t& value : input) {
                value = value >= 128 ? 255 : 0;
            }
        } else {
            if (frame_number >= static_cast<uint64_t>(options.demo_frames)) {
                break;
            }
            make_demo_frame(input, width, height,
                            static_cast<int>(frame_number));
        }

        if (!launch_frame(burn_ns, burn_ns != 0)) {
            std::fprintf(stderr,
                         "Frame %llu activity-path verification failed.\n",
                         static_cast<unsigned long long>(frame_number));
            exit_status = 1;
            break;
        }
        if (!placement_invariant(false)) {
            std::fprintf(stderr,
                         "Frame %llu lost the per-launch warp-slot placement invariant.\n",
                         static_cast<unsigned long long>(frame_number));
            exit_status = 1;
            break;
        }

        size_t changed_slots = 0;
        for (size_t i = 0; i < pixels; ++i) {
            changed_slots +=
                placement[i].smid != initial_placement[i].smid ||
                placement[i].warpid != initial_placement[i].warpid;
        }
        if (changed_slots != 0) {
            ++remapped_frames;
            if (!mapping_warning_printed) {
                std::fprintf(stderr,
                             "Notice: frame %llu remapped %zu/%zu logical warp slots "
                             "relative to the initial launch. Coverage still passed; "
                             "logical pixels are not stable physical locations.\n",
                             static_cast<unsigned long long>(frame_number),
                             changed_slots, pixels);
                mapping_warning_printed = true;
            }
        }

        if (options.raw_output) {
            if (std::fwrite(output.data(), 1, pixels, stdout) != pixels ||
                std::fflush(stdout) != 0) {
                std::perror("Failed to write raw GPU output frame");
                exit_status = 1;
                break;
            }
        } else if (!draw_terminal(output, width, height, first,
                                  options.no_ansi)) {
            std::perror("Failed to write terminal output");
            exit_status = 1;
            break;
        }
        first = false;
        ++frame_number;
        deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            frame_period);
        std::this_thread::sleep_until(deadline);
    }

    if (raw && raw != stdin && std::fclose(raw) != 0) {
        std::perror("Failed to close raw input");
        exit_status = 1;
    }
    if (!options.raw_output && !options.no_ansi) {
        if (std::fputs("\x1b[0m\n", stdout) == EOF || std::fflush(stdout) != 0) {
            std::perror("Failed to finalize terminal output");
            exit_status = 1;
        }
    }
    std::fprintf(stderr,
                 "Rendered %llu frames; physical mapping changed on %zu frames.\n",
                 static_cast<unsigned long long>(frame_number), remapped_frames);

    cleanup();
    return exit_status;
}
