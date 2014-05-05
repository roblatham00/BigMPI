#include "bigmpi_impl.h"

/* There are different ways to implement large-count reductions.
 * The fully general and most correct way to do it is with user-defined
 * reductions, which are required to do reductions on user-defined types,
 * and to parse the datatype inside of a user-defined operation.
 * However, this appear is likely to lead to vastly reduced performance.
 *
 * A less general and potentially unsafe way to implement large-count
 * reductions is to chop them up into multiple messages.
 * This may formally violate the MPI standard, but since BigMPI is
 * not part of the standard, we are going to do it in the name
 * of performance and implementation simplicity. */

int MPIX_Reduce_x(const void *sendbuf, void *recvbuf, MPI_Count count,
                  MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    if (likely (count <= bigmpi_int_max )) {
        return MPI_Reduce(sendbuf, recvbuf, (int)count, datatype, op, root, comm);
    } else {
        int c = (int)(count/bigmpi_int_max);
        int r = (int)(count%bigmpi_int_max);
        if (sendbuf==MPI_IN_PLACE) {
            int commrank;
            MPI_Comm_rank(comm, &commrank);

            for (int i=0; i<c; i++) {
                MPI_Reduce(commrank==root ? MPI_IN_PLACE : &recvbuf[i*bigmpi_int_max],
                           &recvbuf[i*bigmpi_int_max],
                           bigmpi_int_max, datatype, op, root, comm);
            }
            MPI_Reduce(commrank==root ? MPI_IN_PLACE : &recvbuf[c*bigmpi_int_max],
                       &recvbuf[c*bigmpi_int_max],
                       r, datatype, op, root, comm);
        } else {
            for (int i=0; i<c; i++) {
                MPI_Reduce(&sendbuf[i*bigmpi_int_max], &recvbuf[i*bigmpi_int_max],
                           bigmpi_int_max, datatype, op, root, comm);
            }
            MPI_Reduce(&sendbuf[c*bigmpi_int_max], &recvbuf[c*bigmpi_int_max],
                       r, datatype, op, root, comm);
        }
    }
    return MPI_SUCCESS;
}

int MPIX_Allreduce_x(const void *sendbuf, void *recvbuf, MPI_Count count,
                     MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    if (likely (count <= bigmpi_int_max )) {
        return MPI_Allreduce(sendbuf, recvbuf, (int)count, datatype, op, comm);
    } else {
        int c = (int)(count/bigmpi_int_max);
        int r = (int)(count%bigmpi_int_max);
        if (sendbuf==MPI_IN_PLACE) {
            for (int i=0; i<c; i++) {
                MPI_Allreduce(MPI_IN_PLACE, &recvbuf[i*bigmpi_int_max],
                              bigmpi_int_max, datatype, op, comm);
            }
            MPI_Allreduce(MPI_IN_PLACE, &recvbuf[c*bigmpi_int_max],
                          r, datatype, op, comm);
        } else {
            for (int i=0; i<c; i++) {
                MPI_Allreduce(&sendbuf[c*bigmpi_int_max], &recvbuf[i*bigmpi_int_max],
                              bigmpi_int_max, datatype, op, comm);
            }
            MPI_Allreduce(&sendbuf[c*bigmpi_int_max], &recvbuf[c*bigmpi_int_max],
                          r, datatype, op, comm);
        }
    }
    return MPI_SUCCESS;
}

/* MPI-3 Section 5.10
 * Advice to implementers:
 * The MPI_REDUCE_SCATTER_BLOCK routine is functionally equivalent to:
 * an MPI_REDUCE collective operation with count equal to recvcount*n,
 * followed by an MPI_SCATTER with sendcount equal to recvcount. */

/* The previous statement is untrue when sendbuf=MPI_IN_PLACE so we
 * are forced to buffer even in the in-place case. */

int MPIX_Reduce_scatter_block_x(const void *sendbuf, void *recvbuf, MPI_Count recvcount,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    if (likely (recvcount <= bigmpi_int_max )) {
        return MPI_Reduce_scatter_block(sendbuf, recvbuf, (int)recvcount, datatype, op, comm);
    } else {
        int root = 0;

        int commsize;
        MPI_Comm_size(comm, &commsize);
        MPI_Count sendcount = recvcount * commsize;

        MPI_Aint lb /* unused */, extent;
        MPI_Type_get_extent(datatype, &lb, &extent);
        MPI_Aint buf_size = (MPI_Aint)sendcount * extent;

        void * tempbuf = NULL;
        MPI_Alloc_mem(buf_size, MPI_INFO_NULL, &tempbuf);
        if (tempbuf==NULL) { MPI_Abort(comm, 1); }

        MPIX_Reduce_x(sendbuf==MPI_IN_PLACE ? recvbuf : sendbuf,
                      tempbuf, sendcount, datatype, op, root, comm);
        MPIX_Scatter_x(tempbuf, recvcount, datatype, recvbuf, recvcount, datatype, root, comm);

        MPI_Free_mem(&tempbuf);
    }
    return MPI_SUCCESS;
}