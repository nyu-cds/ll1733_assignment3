from itertools import repeat
from itertools import chain
from itertools import combinations
from itertools import permutations
from itertools import compress
from itertools import starmap

def whetherRepeat(a,b): 
    return a!=b

def zbits(n,k):

    a= list(repeat(0,k))  ##consisted of k-bit zeros
    b=list(repeat(1,n-k)) ## the rests are "n-k"-bit ones
    c=list(chain(a,b)) ## make a list of k-bit zeros and "n-k"-bit ones           
    d=sorted(list(permutations(c,n))) ## permutate this list such that all
                                    ## binary strings of length n that contain
                                    ## k zero bits are generated
   
    bools=[]
    for i in range(len(d)-1):    ## during the permutation process, many repeated items
      bools.append(whetherRepeat(d[i],d[i+1]))## are generated, thus need a boolean list
                                              ## to filter out those repeated ones
        
    bools.append(True)##since the last item is always different than the first one, the boolean value has to be true
        
    e= list(compress(d,bools)) ## filter out repeated items

    x=[]
    for i in e:
        t=""
        for j in range(len(i)):
            t+=str(i[j])
        x.append(t)

    y={i for i in x}
    return y
    


