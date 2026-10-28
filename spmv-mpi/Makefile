CC     = gcc
MPICC ?= mpicc
FLAG   = -O3 -std=gnu99 -I./include -Wno-unused-result -Wno-write-strings
LDFLAG = -O3
LIBS   = -lm

SRC = src

# The two implementations you submit, one binary each. Overridable to build
# someone else's file:
#   make spmv_mpi_sparse SPARSE_SRC=../solution/spmv_mpi_sparse.c
# The object name is derived from the source path, so pointing a target at a
# different file never links a stale object from the previous one.
ALLGATHER_SRC ?= $(SRC)/spmv_mpi_allgather.c
SPARSE_SRC    ?= $(SRC)/spmv_mpi_sparse.c

# Headers carry real code here, so rebuild when any of them changes.
HEADERS = $(wildcard include/*.h)

all: spmv spmv_csr spmv_mpi_allgather spmv_mpi_sparse

# ---- sequential baselines (plain gcc) ----
spmv: $(SRC)/spmv.o $(SRC)/mmio.o
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

spmv_csr: $(SRC)/spmv_csr.o $(SRC)/mmio.o
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

%.o: %.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $<

# ---- MPI versions (mpicc, objects kept separate as *.mpi.o) ----
HARNESS_OBJ = $(SRC)/main_mpi.mpi.o $(SRC)/mmio.mpi.o

spmv_mpi_allgather: $(HARNESS_OBJ) $(ALLGATHER_SRC:.c=.mpi.o)
	$(MPICC) $(LDFLAG) -o $@ $^ $(LIBS)

spmv_mpi_sparse: $(HARNESS_OBJ) $(SPARSE_SRC:.c=.mpi.o)
	$(MPICC) $(LDFLAG) -o $@ $^ $(LIBS)

%.mpi.o: %.c $(HEADERS)
	$(MPICC) -o $@ -c $(FLAG) $<

.PHONY: clean all
clean:
	find ./ -name "*.o" -delete
	rm -f spmv spmv_csr spmv_mpi_allgather spmv_mpi_sparse
