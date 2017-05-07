
from pyspark import SparkContext
from operator import add
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
    f=open("numbers",'w')
    for i in range(1,1001):##write nums from 1 to 1000 to a write
        f.write(str(i)+"\n")
    f.close()
    sc = SparkContext("local", "averagesquareroots")
            
    text = sc.textFile('numbers')

    nums =text.flatMap(splitter)##parsing to get a list of numbers
    
    nums_squared = nums.map(lambda x: int(x)**(1/2))### process each number via map to get its square root
    print (nums_squared.fold(0,add)/1000)### use fold function to add these squared values together, and then get average


