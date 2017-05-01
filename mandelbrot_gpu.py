# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    """
    The code first obtains the starting x and y coordinates using
    cuda.grid() function.
    
    Then it obtains the ending values of x and y, namely the edges for
    the image (image.shape[0], image.shape[1]).

    Next it finds the step values for x and y,namely the production of gridDim and blockDim.

    Finally, it uses the derived "start", "end", and "step" values to construct the for
    loop to start iteration.
    """
    xstart, ystart = cuda.grid(2)  ## starting values of x and y


    height,width=image.shape[0],image.shape[1]     ## boundaried of the image, namely ending values for y and x
   
    xstep=cuda.gridDim.x*cuda.blockDim.x  ## the shape of the grid in x-axis designates the step for x
    ystep=cuda.gridDim.y*cuda.blockDim.y  ## the shape of the grid in y-axis designates the step for y 
    
   
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(xstart,width,xstep):  ## start, end, step
        real = min_x + x * pixel_size_x
        for y in range(ystart,height,ystep): ## start, end , step
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters)
        
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)
    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()
