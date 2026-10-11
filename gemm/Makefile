CC     = gcc
MPICC  = mpicc
FLAG   = -O3 -std=gnu99 -g -I./include -Wno-unused-result
LDFLAG = -O3 -g
LIBS   = -lm

# Each stage is built with exactly the flags it is allowed to use, so an
# OpenMP pragma in the tiled file or an AVX intrinsic in the OpenMP file is a
# build error rather than a blurred roofline point. The harness itself is
# always built with OpenMP (it runs the reference check in parallel).
OMPFLAG  = -fopenmp
SIMDFLAG = -march=native

# make gemm_naive PROFILE=1 -> gprof-instrumented (-pg); -g is on regardless,
# so perf record / perf report have symbols without it.
ifdef PROFILE
FLAG   += -pg
LDFLAG += -pg
endif

SRC = src

# The four files you submit, one binary each. Overridable to build someone
# else's file:
#   make gemm_omp OMP_SRC=path/to/gemm_omp.c
# The object name is derived from the source path, so pointing a target at a
# different file never links a stale object from the previous one.
NAIVE_SRC ?= $(SRC)/gemm_naive.c
TILED_SRC ?= $(SRC)/gemm_tiled.c
OMP_SRC   ?= $(SRC)/gemm_omp.c
SIMD_SRC  ?= $(SRC)/gemm_simd.c
MPI_SRC   ?= $(SRC)/gemm_mpi.c

# Which of your Part 1 kernels SUMMA calls for the local multiply:
#   make gemm_mpi LOCAL_GEMM=simd      (default; also: omp, tiled)
LOCAL_GEMM ?= simd
ifeq ($(LOCAL_GEMM),simd)
LOCAL_OBJ = $(SIMD_SRC:.c=.simd.o)
else ifeq ($(LOCAL_GEMM),omp)
LOCAL_OBJ = $(OMP_SRC:.c=.omp.o)
else ifeq ($(LOCAL_GEMM),tiled)
LOCAL_OBJ = $(TILED_SRC:.c=.serial.o)
else
$(error LOCAL_GEMM must be simd, omp or tiled)
endif

# Headers carry real code here, so rebuild when any of them changes.
HEADERS = $(wildcard include/*.h)

all: gemm_naive gemm_tiled gemm_omp gemm_simd gemm_mpi stream

# ---- the single-node harness, shared by the four Part 1 binaries ----
$(SRC)/main_gemm.o: $(SRC)/main_gemm.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $(OMPFLAG) $<

gemm_naive: $(SRC)/main_gemm.o $(NAIVE_SRC:.c=.serial.o)
	$(CC) $(LDFLAG) $(OMPFLAG) -o $@ $^ $(LIBS)

gemm_tiled: $(SRC)/main_gemm.o $(TILED_SRC:.c=.serial.o)
	$(CC) $(LDFLAG) $(OMPFLAG) -o $@ $^ $(LIBS)

gemm_omp: $(SRC)/main_gemm.o $(OMP_SRC:.c=.omp.o)
	$(CC) $(LDFLAG) $(OMPFLAG) -o $@ $^ $(LIBS)

gemm_simd: $(SRC)/main_gemm.o $(SIMD_SRC:.c=.simd.o)
	$(CC) $(LDFLAG) $(OMPFLAG) -o $@ $^ $(LIBS)

# ---- the SUMMA harness: your gemm_mpi.c plus one of your Part 1 kernels ----
$(SRC)/main_summa.o: $(SRC)/main_summa.c $(HEADERS)
	$(MPICC) -o $@ -c $(FLAG) $(OMPFLAG) $<

gemm_mpi: $(SRC)/main_summa.o $(MPI_SRC:.c=.mpi.o) $(LOCAL_OBJ)
	$(MPICC) $(LDFLAG) $(OMPFLAG) -o $@ $^ $(LIBS)

%.serial.o: %.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $<

%.omp.o: %.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $(OMPFLAG) $<

%.simd.o: %.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $(OMPFLAG) $(SIMDFLAG) $<

%.mpi.o: %.c $(HEADERS)
	$(MPICC) -o $@ -c $(FLAG) $(OMPFLAG) $<

# ---- the memory-bandwidth ceiling ----
stream: tools/stream.c
	$(CC) $(FLAG) $(OMPFLAG) -o $@ $< $(LIBS)

.PHONY: all clean
clean:
	find ./ -name "*.o" -delete
	rm -f gemm_naive gemm_tiled gemm_omp gemm_simd gemm_mpi stream gmon.out
