from pyspark import SparkContext

def product(a,b):
    """
    return the product of inputs
    """
    return a*b
    
if __name__ == '__main__':
	sc = SparkContext("local", "primes")
	# Create an RDD of numbers from 1 to 1,001 (inclusive)
	nums = sc.parallelize(range(1,1001))
         
	# Compute the production of these numbers in the RDD via fold
	print (nums.fold(1,product))
