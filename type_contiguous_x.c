#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#include <mpi.h>

#define BIGMPI_MAX 2147483648

#define MPI_ASSERT(e)  \
        ((void) ((e) ? 0 : MPI_Abort(MPI_COMM_WORLD,1) ))

/*
 * Synopsis
 *
 * int MPIX_Type_contiguous_x(MPI_Count count,
 *                            MPI_Datatype   oldtype,
 *                            MPI_Datatype * newtype)
 *                         
 *  Input Parameters
 *
 *   count             replication count (nonnegative integer)
 *   oldtype           old datatype (handle)
 *
 * Output Parameter
 *
 *   newtype           new datatype (handle)
 *
 */
int MPIX_Type_contiguous_x(MPI_Count count, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int rc = MPI_SUCCESS;

    MPI_Count c = count/BIGMPI_MAX;
    MPI_Count r = count%BIGMPI_MAX;

    MPI_Datatype chunk;
    rc = MPI_Type_contiguous(BIGMPI_MAX, oldtype, &chunk);
    MPI_ASSERT(rc==MPI_SUCCESS);

    MPI_Datatype chunks;
    rc = MPI_Type_contiguous(c, chunk, &chunks);
    MPI_ASSERT(rc==MPI_SUCCESS);

    MPI_Datatype remainder;
    rc = MPI_Type_contiguous(r, oldtype, &remainder);
    MPI_ASSERT(rc==MPI_SUCCESS);

    int typesize;
    rc = MPI_Type_size(oldtype, &typesize);

    MPI_Aint remdisp                   = (MPI_Aint)c*BIGMPI_MAX*typesize; /* must explicit-cast to avoid overflow */
    int array_of_blocklengths[2]       = {1,1};
    MPI_Aint array_of_displacements[2] = {0,remdisp};
    MPI_Datatype array_of_types[2]     = {chunks,remainder};
    rc = MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, newtype);
    MPI_ASSERT(rc==MPI_SUCCESS);

    rc = MPI_Type_commit(newtype);
    MPI_ASSERT(rc==MPI_SUCCESS);

    rc = MPI_Type_free(&chunk);
    MPI_ASSERT(rc==MPI_SUCCESS);

    rc = MPI_Type_free(&chunks);
    MPI_ASSERT(rc==MPI_SUCCESS);

    rc = MPI_Type_free(&remainder);
    MPI_ASSERT(rc==MPI_SUCCESS);

    return MPI_SUCCESS;
}

