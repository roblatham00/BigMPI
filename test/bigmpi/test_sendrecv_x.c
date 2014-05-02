#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <strings.h>

#include <mpi.h>
#include "bigmpi.h"

#ifdef BIGMPI_MAX_INT
const MPI_Count test_int_max = BIGMPI_MAX_INT;
#else
#include <limits.h>
const MPI_Count test_int_max = INT_MAX;
#endif

/* Yes, it is technically unsafe to cast MPI_Count to MPI_Aint or size_t without checking,
 * given that MPI_Count might be 128b and MPI_Aint and size_t might be 64b, but BigMPI
 * does not aspire to support communication of more than 8 EiB messages at a time.
 * If your system has more than 8 EiB of memory, please contact me, as I would like to use it :-) */

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size<2) {
        printf("Use 2 or more processes. \n");
        MPI_Finalize();
        return 1;
    }

    int m = (argc > 1) ? atoi(argv[1]) : 2;
    MPI_Count n = m*test_int_max+17777;

    char * buf = NULL;

    MPI_Alloc_mem((MPI_Aint)n, MPI_INFO_NULL, &buf);

    memset(buf, rank, (size_t)n);

    for (int r = 1; r < size; r++) {

        /* pairwise communication */
        if (rank==r) {
            MPIX_Send_x(buf, n, MPI_CHAR, 0 /* dst */, r /* tag */, MPI_COMM_WORLD);
        }
        else if (rank==0) {
            MPIX_Recv_x(buf, n, MPI_CHAR, r /* src */, r /* tag */, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* correctness verification */
            size_t errors = 0;
            for (size_t i = 0; i < (size_t)n; i++) {
                errors += (buf[i] != (unsigned char)r );
            }
            if (errors > 0) {
                printf("There were %zu errors!", errors);
                MPI_Abort(MPI_COMM_WORLD, (int)errors);
            }
        }
    }

    MPI_Free_mem(buf);

    if (rank==0) {
        printf("SUCCESS\n");
    }

    MPI_Finalize();

    return 0;
}
