from pyspark import SparkContext
from operator import add
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
	sc = SparkContext("local", "wordcount")
	
	text = sc.textFile('pg2701.txt')
	words = text.flatMap(splitter)
	words_mapped = words.map(lambda x: (x,1))

	"""
        distinct(self)  Return a new RDD containing the distinct
        elements in this RDD.

	count(self)  Return the number of elements in this RDD.

        Thus, the combination of these two functions can give us
        the number of distinct words in the input text.      
	"""

        
	print (words_mapped.distinct().count())
