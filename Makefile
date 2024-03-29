# only tested with clang-16, gcc doesn't work
CC := clang-16

.PHONY: all clean

all: mm-bench mm-bench-ebm

mm-bench: mm-bench.c
	$(CC) -O3 -march=armv8-a -static $^ -o $@

mm-bench-ebm: mm-bench.c
	$(CC) -O3 -march=armv8-a -static -DEBM $^ -o $@

clean:
	rm -f mm-bench mm-bench-ebm
