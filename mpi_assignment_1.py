import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank %2== 0:
        print ("Hello")
        
else:
        print ("Goodbye from process "+str(rank))
        
