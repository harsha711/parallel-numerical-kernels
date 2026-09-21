CC     = gcc
FLAG   = -O3 -std=gnu99 -fopenmp -I./include -Wno-unused-result -Wno-write-strings
LDFLAG = -O3 -fopenmp
LIBS   = -lm

# The SIMD kernels need AVX2 + FMA. -march=native picks them up on any machine
# new enough to run this assignment; it is separated out so the OpenMP targets
# still build on a machine without AVX2.
SIMDFLAG = -march=native

SRC = src

# The four implementations you submit, one binary each. Overridable to build
# someone else's file:
#   make spmv_omp_ell OMP_ELL_SRC=../solution/spmv_omp_ell.c
# The object name is derived from the source path, so pointing a target at a
# different file never links a stale object from the previous one.
OMP_CSR_SRC  ?= $(SRC)/spmv_omp_csr.c
OMP_ELL_SRC  ?= $(SRC)/spmv_omp_ell.c
SIMD_CSR_SRC ?= $(SRC)/spmv_simd_csr.c
SIMD_ELL_SRC ?= $(SRC)/spmv_simd_ell.c

# Headers carry real code here, so rebuild when any of them changes.
HEADERS = $(wildcard include/*.h)

all: spmv spmv_csr spmv_omp_csr spmv_omp_ell spmv_simd_csr spmv_simd_ell

# ---- sequential baselines (no OpenMP, no SIMD) ----
spmv: $(SRC)/spmv.o $(SRC)/mmio.o
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

spmv_csr: $(SRC)/spmv_csr.o $(SRC)/mmio.o
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

%.o: %.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $<

# ---- the harness, shared by all four targets ----
HARNESS_OBJ = $(SRC)/main_omp.o $(SRC)/mmio.o

spmv_omp_csr: $(HARNESS_OBJ) $(OMP_CSR_SRC:.c=.o)
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

spmv_omp_ell: $(HARNESS_OBJ) $(OMP_ELL_SRC:.c=.o)
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

# ---- SIMD versions (same harness, kernels built with -march=native) ----
spmv_simd_csr: $(HARNESS_OBJ) $(SIMD_CSR_SRC:.c=.simd.o)
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

spmv_simd_ell: $(HARNESS_OBJ) $(SIMD_ELL_SRC:.c=.simd.o)
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

%.simd.o: %.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $(SIMDFLAG) $<

.PHONY: clean all
clean:
	find ./ -name "*.o" -delete
	rm -f spmv spmv_csr spmv_omp_csr spmv_omp_ell spmv_simd_csr spmv_simd_ell
