#include "bigmpi_impl.h"

/*
 * Synopsis
 *
 * a version of MPI_Type_create_hvector, except the array_of_blocklengths can
 * be larger than 32 bits
 *
 * int MPIX_Type_create_hvector_x(MPI_Count count,
 *                                MPI_Count array_of_blocklengths[],
 *                                MPI_Aint array_of_displacements[],
 *                                MPI_Datatype   oldtype,
 *                                MPI_Datatype * newtype)
 *
 *  Input Parameters
 *
 *   count                   number of blocks -- also number of entries in
 *                           array_of_displacements and array_of_blocklengths
 *                           (non-negative integer)
 *
 *   array_of_blocklengths   number of elements in each block (array of
 *                           non-negative integers)
 *
 *   array_of_displacements  byte displacement of each block (array of
 *                           integers)
 *
 *   oldtype                 old datatype (handle)
 *
 * Output Parameter
 *
 *   newtype           new datatype (handle)
 *
 */
int MPIX_Type_create_hvector_x(int count,
	MPI_Count array_of_blocklengths[], MPI_Aint array_of_displacements[],
	MPI_Datatype oldtype, MPI_Datatype * newtype)
{
    int i, ret;
    MPI_Datatype *types;
    int *blocklens;

    /* The count has to fit into MPI_Aint for BigMPI to work. */
    assert(count<bigmpi_count_max);

    types = malloc(count*sizeof(*types));
    blocklens = malloc(count*sizeof(*blocklens));

    for (i=0; i<count; i++) {
	blocklens[i] = 1;
	MPIX_Type_contiguous_x(array_of_blocklengths[i], oldtype,  &(types[i]));
    }

    ret = MPI_Type_create_struct(count, blocklens, array_of_displacements,
	    types, newtype);

    for (i=0; i<count; i++)
	MPI_Type_free(&(types[i]));

    free(types);
    free(blocklens);

    return ret;
}


/*
 * Synopsis
 *
 * This function inverts MPIX_Type_contiguous_x, i.e. it provides
 * the original arguments for that call so that we know how many
 * built-in types are in the user-defined datatype.
 *
 * This function is primary used inside of BigMPI and does not
 * correspond to an MPI function, so we do avoid the use of the
 * MPIX namespace.
 *
 * int BigMPI_Decode_contiguous_x(MPI_Datatype   intype,
 *                                MPI_Count    * count,
 *                                MPI_Datatype * basetype)
 *
 *  Input Parameters
 *
 *   newtype           new datatype (handle)
 *
 * Output Parameter
 *
 *   count             replication count (nonnegative integer)
 *   oldtype           old datatype (handle)
 *
 */
int BigMPI_Decode_contiguous_x(MPI_Datatype intype, MPI_Count * count, MPI_Datatype * basetype)
{
    int nint, nadd, ndts, combiner;

    /* Step 1: Decode the type_create_struct call. */

    MPI_Type_get_envelope(intype, &nint, &nadd, &ndts, &combiner);
    assert(combiner==MPI_COMBINER_STRUCT || combiner==MPI_COMBINER_VECTOR);
#ifdef BIGMPI_AVOID_TYPE_CREATE_STRUCT
    if (combiner==MPI_COMBINER_VECTOR) {
        assert(nint==3);
        assert(nadd==0);
        assert(ndts==1);

        int cbs[3]; /* {count,blocklength,stride} */
        MPI_Datatype vbasetype[1];
        MPI_Type_get_contents(intype, 3, 0, 1, cbs, NULL, vbasetype);
        MPI_Count a = cbs[0];   /* count */
        MPI_Count b = cbs[1];   /* blocklength */
        assert(cbs[1]==cbs[2]); /* blocklength==stride */

        *count = a*b;
        *basetype = vbasetype[0];
        return MPI_SUCCESS;
    }
#else
    assert(combiner==MPI_COMBINER_STRUCT);
#endif
    assert(nint==3);
    assert(nadd==2);
    assert(ndts==2);

    int cnbls[3]; /* {count, blocklengths[]} */
    MPI_Aint displacements[2]; /* {0,remdisp} */
    MPI_Datatype types[2]; /* {chunks,remainder} */;
    MPI_Type_get_contents(intype, 3, 2, 2, cnbls, displacements, types);
    assert(cnbls[0]==2);
    assert(cnbls[1]==1);
    assert(cnbls[2]==1);
    assert(displacements[0]==0);

    /* Step 2: Decode the type_vector call. */

    MPI_Type_get_envelope(types[0], &nint, &nadd, &ndts, &combiner);
    assert(combiner==MPI_COMBINER_VECTOR);
    assert(nint==3);
    assert(nadd==0);
    assert(ndts==1);

    int cbs[3]; /* {count,blocklength,stride} */
    MPI_Datatype vbasetype[1];
    MPI_Type_get_contents(types[0], 3, 0, 1, cbs, NULL, vbasetype);
    assert(/* blocklength = */ cbs[1]==bigmpi_int_max);
    assert(/* stride = */ cbs[2]==bigmpi_int_max);

    /* chunk count - see above */
    MPI_Count c = cbs[0];

    /* Step 3: Decode the type_contiguous call. */

    MPI_Type_get_envelope(types[1], &nint, &nadd, &ndts, &combiner);
    assert(combiner==MPI_COMBINER_CONTIGUOUS);
    assert(nint==1);
    assert(nadd==0);
    assert(ndts==1);

    int ccc[1]; /* {count} */
    MPI_Datatype cbasetype[1];
    MPI_Type_get_contents(types[1], 1, 0, 1, ccc, NULL, cbasetype);

    /* remainder - see above */
    MPI_Count r = ccc[0];

    /* The underlying type of the vector and contig types must match. */
    assert(cbasetype[0]==vbasetype[0]);
    *basetype = cbasetype[0];

    /* This should not overflow because everything is already MPI_Count type. */
    *count = c*bigmpi_int_max+r;

    return MPI_SUCCESS;
}
