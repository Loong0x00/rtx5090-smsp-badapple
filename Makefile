NVCC ?= nvcc
TARGETS := smsp_badapple smsp_allocation_probe
NVCCFLAGS := -O3 -std=c++17 -arch=sm_120 -lineinfo

.PHONY: all clean ptx

all: $(TARGETS)

smsp_badapple: smsp_badapple.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

smsp_allocation_probe: smsp_allocation_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

ptx: smsp_badapple.ptx

smsp_badapple.ptx: smsp_badapple.cu
	$(NVCC) -O3 -std=c++17 -arch=compute_120 -ptx $< -o $@

clean:
	rm -f $(TARGETS) smsp_badapple.ptx
