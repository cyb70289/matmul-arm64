CXX := clang++-16
ARCH := $(shell uname -p)

.PHONY: clean

mm-bench: mm-bench.cc mm.cc
	$(CXX) -std=c++17 -O3 -DNDEBUG -march=armv8-a -pthread $^ -o $@

clean:
	rm -f mm-bench
