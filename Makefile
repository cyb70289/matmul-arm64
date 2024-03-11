# only tested with clang-16, gcc doesn't work
CXX = clang++-16

.PHONY: clean bench bench-all profile profile-all test

mm-bench: mm-bench.cc mm.cc mm-panel.S mm-tile.S
	$(CXX) -std=c++17 -O3 -march=armv8-a -static $^ -o $@

bench: mm-bench
	./mm-bench

bench-all: mm-bench
	./mm-bench all

profile: mm-bench
	perf stat -e L1D_CACHE_REFILL,L1D_CACHE,L2D_CACHE_REFILL,L2D_CACHE \
          -e BR_MIS_PRED_RETIRED,BR_RETIRED,ASE_SPEC,instructions,cycles \
          ./mm-bench 2>&1 | sed 's/ \+(.*%)$$//'

profile-all: mm-bench
	for bm in $$(./mm-bench list); do \
    perf stat -e L1D_CACHE_REFILL,L1D_CACHE,L2D_CACHE_REFILL,L2D_CACHE \
              -e BR_MIS_PRED_RETIRED,BR_RETIRED,ASE_SPEC,instructions,cycles \
              ./mm-bench $${bm} 2>&1 | sed 's/ \+(.*%)$$//'; \
    done

test: mm-bench
	./mm-bench test

clean:
	rm -f mm-bench
