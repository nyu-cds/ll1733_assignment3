import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
y=0
l= numpy.zeros(1)

if rank == 0:
        while (True):## keep asking for input until the format of the input is correct
                x=input("enter a number\n")
                try:
                        y=int(x)
                except ValueError:
                        print ("please enter an integer!")
                        continue
                else:
                        if y>=100:
                                print ("The number must be less than 100!")
                                continue
                        break
        l=numpy.array([y])
        comm.Send(l, dest=1)
        comm.Recv(l,source=size-1)
        print (l[0])
else:       
        comm.Recv(l, source=rank-1)
        l*=rank
        comm.Send(l, dest=(rank+1)%size) 
        




        
