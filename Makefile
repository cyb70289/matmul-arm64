# only tested with clang-16, gcc doesn't work
CXX := clang++-16
ARCH := $(shell uname -p)

mm-bench:

.PHONY: test clean bench bench-all profile profile-all

test: mm-bench
	./mm-bench test

clean:
	rm -f mm-bench

bench: mm-bench
	./mm-bench

bench-all: mm-bench
	./mm-bench all

ifeq ($(ARCH),aarch64)

PERF_EVENTS = L1D_CACHE_REFILL,L1D_CACHE,ASE_SPEC

mm-bench: mm-bench.cc mm.cc mm-panel.S mm-tile.S
	$(CXX) -std=c++17 -O3 -march=armv8-a $^ -o $@

profile: mm-bench
	perf stat -e $(PERF_EVENTS),instructions,cycles,task-clock \
	    ./mm-bench 2>&1 | sed 's/ \+(.*%)$$//'

profile-all: mm-bench
	for bm in $$(./mm-bench list); do \
	    perf stat -e $(PERF_EVENTS),instructions,cycles,task-clock \
	        ./mm-bench $${bm} 2>&1 | sed 's/ \+(.*%)$$//'; \
	done

endif  # aarch64

ifeq ($(ARCH),x86_64)

PERF_EVENTS = L1D.REPLACEMENT,FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE

# XXX: march=cascadelake to enable ymm16~31
mm-bench: mm-bench.cc mm.cc mm-avx2.cc
	$(CXX) -std=c++17 -O3 -march=cascadelake $^ -o $@

profile: mm-bench
	perf stat -e $(PERF_EVENTS),instructions,cycles,task-clock ./mm-bench

profile-all: mm-bench
	for bm in $$(./mm-bench list); do \
	    perf stat -e $(PERF_EVENTS),instructions,cycles,task-clock \
	    ./mm-bench $${bm}; \
	done

endif  # x86_64
