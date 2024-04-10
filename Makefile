# only tested with clang-16, gcc doesn't work
CXX := clang++-16
ARCH := $(shell uname -p)

.PHONY: clean bench bench-all bench-onednn profile profile-all test

mm-bench: mm-bench.cc mm.cc mm-panel.S mm-tile.S
	$(CXX) -std=c++17 -O3 -DNDEBUG -march=armv8-a -static $^ -o $@

bench: mm-bench
	./mm-bench

bench-all: mm-bench
	./mm-bench all

profile: mm-bench
	perf stat \
	    -e L1D_CACHE_REFILL,L1D_CACHE,L2D_CACHE_REFILL,L2D_CACHE \
	    -e BR_MIS_PRED_RETIRED,BR_RETIRED,ASE_SPEC,instructions,cycles \
	    ./mm-bench 2>&1 | sed 's/ \+(.*%)$$//'

profile-all: mm-bench
	for bm in $$(./mm-bench list); do \
	    perf stat \
	        -e L1D_CACHE_REFILL,L1D_CACHE,L2D_CACHE_REFILL,L2D_CACHE \
	        -e BR_MIS_PRED_RETIRED,BR_RETIRED,ASE_SPEC,instructions,cycles \
	        ./mm-bench $${bm} 2>&1 | sed 's/ \+(.*%)$$//'; \
	done

test: mm-bench
	./mm-bench test

clean:
	rm -f mm-bench onednn-bench blis-bench llamafile-bench

#################################### onednn ####################################
# - build acl (arm only)
#   $ git clone https://github.com/ARM-software/ComputeLibrary --depth=1
#   $ cmake -S ComputeLibrary/ -B ComputeLibrary/build
#   $ make -C ComputeLibrary/build -j32
# - build onednn
#   $ git clone https://github.com/oneapi-src/onednn --depth=1
#   $ ACL_ROOT_DIR=../ComputeLibrary/ cmake -S onednn -B onednn/build \
#       -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_DOC=OFF \
#       -DDNNL_AARCH64_USE_ACL=ON -DCMAKE_INSTALL_PREFIX=onednn/build
#   $ make -C onednn/build install -j32

# lib path
DNNL_LIBS := -ldnnl
DNNL_LDFLAGS := -L./onednn/build/lib
DNNL_LDLIB_DIR := ./onednn/build/lib
ifeq ($(ARCH),aarch64)
    DNNL_LIBS += -larm_compute -larm_compute_graph
    DNNL_LDFLAGS += -L./ComputeLibrary/build
    DNNL_LDLIB_DIR := $(DNNL_LDLIB_DIR):./ComputeLibrary/build
endif

# ensure matrix shapes are only from command line, not environment variable
# e.g., make bench-onednn B=100 M=1024 N=256 K=64
ifneq ($(origin B), command line)
    B :=
endif
ifneq ($(origin M), command line)
    M :=
endif
ifneq ($(origin N), command line)
    N :=
endif
ifneq ($(origin K), command line)
    K :=
endif

onednn-bench: onednn-bench.cc
	g++ -O3 -std=c++17 $^ -o $@ -I./onednn/build/include $(DNNL_LDFLAGS) $(DNNL_LIBS)

bench-onednn: onednn-bench
	OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$(DNNL_LDLIB_DIR) ./onednn-bench $(B) $(M) $(N) $(K)

################################## llamafile ##################################
llamafile-bench: llamafile-bench.cc
	g++ -O3 -std=c++17 $^ -o $@

profile-llamafile: llamafile-bench
	perf stat \
        -e L1D_CACHE_REFILL,L1D_CACHE,L2D_CACHE_REFILL,L2D_CACHE \
        -e BR_MIS_PRED_RETIRED,BR_RETIRED,ASE_SPEC,instructions,cycles \
        ./llamafile-bench 2>&1 | sed 's/ \+(.*%)$$//'
