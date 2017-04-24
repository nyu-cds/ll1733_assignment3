import random
import numpy
import math
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size=comm.Get_size()
x=[[]for i in range(size)]
def s(l):
    return sorted(l)
def parallelSort(l):
    if rank == 0:
        data=[[]for i in range(size)]## prepare space to store sublists
        hi=max(l)
        lo=min(l)
        for i in l: ## distribute each element to different area based on its value
            pointer=min(size-1,int((i-lo)/math.ceil((hi-lo)/size)))
            data[pointer].append(i)

    else:
        data=None

    scatter=comm.scatter(data,root=0) ## scatter subslist to different processes
    ss=s(scatter)
    gather=comm.gather(ss,root=0)  ## gather sorted sublists together

    if rank==0:

        final=[]
        for i in range(len(gather)): ##cancatenate each splitted sorted list into one complete sorted list
            final+=gather[i]
        print ("The sorted list is", final)
        return final

if __name__ == '__main__':
    l=[]
    if rank==0:
        l=numpy.random.randint(0,10000,size=10000)##generate random data
        print ("list is ", l)
    parallelSort(l)
