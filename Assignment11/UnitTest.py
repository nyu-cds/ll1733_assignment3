import parallel_sorter
import numpy
import unittest
from mpi4py import MPI
comm = MPI.COMM_WORLD
import random
import math
size=comm.Get_size()
class Test(unittest.TestCase):
    def test_output_for_different_processes(self):
        rank=comm.Get_rank()
        sample=[i for i in range(100)]
        random.shuffle(sample)
        output=parallel_sorter.parallelSort(sample)
        
        if rank==0:
            correctAnswer=sorted(sample)
            self.assertEqual(output,correctAnswer)
        else:
            self.assertEqual(output,None)

if __name__=='__main__':
    unittest.main()
