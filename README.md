BigMPI
======

Interface to MPI for large messages, i.e. those where the count argument
exceeds `INT_MAX` but is still less than `SIZE_MAX`.
BigMPI is designed for the common case where one has a 64b address
space and is unable to do MPI communication on more than 2^31 elements
despite having sufficient memory to allocate such buffers.
BigMPI does not attempt to support large-counts on systems where
C int and void* are both 32b.

## Motivation

The MPI standard provides a wide range of communication functions that
take a C `int` argument for the element count, thereby limiting this
value to `INT_MAX` or less.
This means that one cannot send, e.g. 3 billion bytes using the `MPI_BYTE`
datatype, or a vector of 5 billion integers using the `MPI_INT` type, as
two examples.
There is a natural workaround using MPI derived datatypes, but this is
a burden on users who today may not be using derived datatypes.

This project aspires to make it as easy as possible to support arbitrarily
large counts (2^63 elements exceeds the local storage compacity of computers 
for the foreseeable future).

This is an example of the code change required to support large counts using
BigMPI:
```
#ifdef BIGMPI
    MPIX_Bcast_x(stuff, large_count /* MPI_Count */, MPI_BYTE, 0, MPI_COMM_WORLD);
#else // cannot use count>INT_MAX
    MPI_Bcast(stuff, not_large_count /* int */, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
```

## Interface
The API follows the pattern of `MPI_Type_size(_x)` in that all BigMPI
functions are identical to their corresponding MPI ones except that
they end with `_x` to indicate that the count arguments have the type
`MPI_Count` instead of `int`.
BigMPI functions use the MPIX namespace because they are not in the
MPI standard.

## Limitations
Even though `MPI_Count` might be 128b, I am only supporting
64b counts (because of `MPI_Aint` limitations and desire to use `size_t`
in my unit tests), so BigMPI is not going to solve your problem if you
want to communicate more than 8 EiB of data in a single message.
Such computers do not exist nor is it likely that they will exist
in the foreseeable future.

BigMPI requires C99.  If your compiler does not support C99, get a
new compiler.

## Supported Functions
I believe that point-to-point, one-sided, broadcast and reductions
are the only functions worth supporting but I added some of the other 
collectives anyways.
 he other collectives clearly aren't scalable because they are going
to move O(nproc*count) data, which is obviously a lot of memory in the
case of e.g. allgather of >2GiB for more than a few dozen procs.
so I will not support these unless >1 users claim it is necessary.
The v-collectives are a pain because one needs to create a new count
vector or do other inefficient things (like implement collectives using
point-to-point).

## Technical details

[MPIX_Type_contiguous_x](https://github.com/jeffhammond/BigMPI/blob/master/src/type_contiguous_x.c)
does the heavy lifting.  It's pretty obvious how it works.
The datatypes engine will turn this into a contiguous datatype internally 
and thus the underlying communication will be efficient.  
MPI implementations need to be count-safe for this to work, but they need
to be count-safe period if the Forum is serious about datatypes being
the solution rather than `MPI_Count` everywhere.

All of the communication functions follow the same pattern, which is
clearly seen in [MPIX_Send_x](https://github.com/jeffhammond/BigMPI/blob/master/src/sendrecv_x.c).
I've optimized for the common case when count is smaller than 2^31 
with a `likely_if` macro to minimize the performance hit of BigMPI
for this more common use case
(hopefully so that users don't insert a branch for this themselves)

The most obvious optimization I can see doing is to implement
`MPIX_Type_contiguous_x` using internals of the MPI implementation 
instead of calling six MPI datatype functions.
I have started implemented this in MPICH already: 
https://github.com/jeffhammond/mpich/tree/type_contiguous_x.

## Authors

* Jeff Hammond
* Andreas Schäfer

## Related

https://svn.mpi-forum.org/trac/mpi-forum-web/ticket/423
