# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 

##Before made any change, the performace data of the orginal file from line_profiler is listed below:

#Total time: 3.77502 s
#File: <ipython-input-2-b142caa8fdc3>
#Function: hypotenuse at line 42

#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#    42                                           def hypotenuse(x,y):
#    43                                               """
#    44                                               Return sqrt(x**2 + y**2) for two arrays, a and b.
#    45                                               x and y must be two-dimensional arrays of the same shape.
#    46                                               """
#    47         1       970808 970808.0     25.7      xx = multiply(x,x)
#    48         1       942689 942689.0     25.0      yy = multiply(y,y)
#    49         1       944512 944512.0     25.0      zz = add(xx, yy)
#    50         1       917015 917015.0     24.3      return sqrt(zz)
#===============================================================

## Part of (because the complete list is too long )the performace data of the orginal file from cProfile is listed below:
#===============================================================
#   1015920 function calls (1015799 primitive calls) in 1.932 seconds

#   Ordered by: internal time

#   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#        2    0.912    0.456    0.920    0.460 calculator.py:19(multiply)
#        1    0.456    0.456    0.460    0.460 calculator.py:6(add)
#        1    0.372    0.372    0.450    0.450 calculator.py:32(sqrt)
#  1000001    0.073    0.000    0.073    0.000 {math.sqrt}
#        2    0.030    0.015    0.030    0.015 {method 'random_sample' of 'mtrand.RandomState' objects}
#        4    0.023    0.006    0.097    0.024 __init__.py:1(<module>)
#     4097    0.017    0.000    0.017    0.000 {range}
#        1    0.006    0.006    0.006    0.006 __init__.py:88(<module>)
#        1    0.003    0.003    1.932    1.932 calculator_test.py:4(<module>)
#        1    0.003    0.003    0.069    0.069 __init__.py:106(<module>)
#        1    0.002    0.002    0.002    0.002 hashlib.py:56(<module>)
#===================================================================

# From the data of cProfile, it is noticeble that the functions "multiply", "add", "sqrt" occupy most of the time.
# So I use numpy's built-in module to rewrite the functions "multiply", "add" and "sqrt".
# The performance data of the improved file is listed below:

#Total time: 0.013691 s
#File: <ipython-input-9-06ccd167d34b>
#Function: hypotenuse at line 30

#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#    30                                           def hypotenuse(x,y):
#    31                                               """
#    32                                               Return sqrt(x**2 + y**2) for two arrays, a and b.
#    33                                               x and y must be two-dimensional arrays of the same shape.
#    34                                               """
#    35         1         3346   3346.0     24.4      xx = multiply(x,x)
#    36         1         2790   2790.0     20.4      yy = multiply(y,y)
#    37         1         3804   3804.0     27.8      zz = add(xx, yy)
#    38         1         3751   3751.0     27.4      return sqrt(zz)
#_______________________________________________________________________________

# Hence, the speedup is calculated as 3.77502s / 0.013691 = 275.73004.

#______________________________________________________________________________

import numpy as np

def add(x,y):
    """
    Add two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    z=np.add(x,y)
    return z


def multiply(x,y):
    """
    Multiply two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    z=np.multiply(x,y)

    return z


def sqrt(x):
    """
    Take the square root of the elements of an arrays using a Python loop.
    """
    z=np.sqrt(x)

    return z


def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = multiply(x,x)
    yy = multiply(y,y)
    zz = add(xx, yy)
    return sqrt(zz)
