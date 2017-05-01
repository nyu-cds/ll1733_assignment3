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
    
    The it obtains the total units of x and y, namely
    the production of gridDim and blockDim.
    
    Next use image's width and height to divide the total units of
    x and that of y correspondingly, and the results are xRange and
    yRange, which represent the number of steps to go from starting x and
    y coordinates to finishing x and y coordinates.
    
    To calculate the values for real and imag, the old "x" and "y" are
    substituted by xoffset and yoffset, which are adjusted based on starting
    indices and unit steps.
    """
    xstart, ystart = cuda.grid(2)  ## starting indices of x and y

    xcapacity=cuda.gridDim.x*cuda.blockDim.x      ##  total units of x
    ycapacity=cuda.gridDim.y*cuda.blockDim.y      ##  total units of y
    
    xRange=int((image.shape[1])/(xcapacity)) ##getting evenly mapping between image and grid for x
    yRange=int((image.shape[0])/(ycapacity))  ##getting evenly mapping between image and grid for y
    
    pixel_size_x = (max_x - min_x) / image.shape[1]
    pixel_size_y = (max_y - min_y) / image.shape[0]
    for x in range(xRange):
        xoffset=xstart+xcapacity*x
        if xoffset>=image.shape[1]:
            break
        real = min_x + xoffset * pixel_size_x
        for y in range(yRange):
            yoffset=ystart+ycapacity*y
            
            if yoffset >=image.shape[0]:
                break
            imag = min_y + yoffset * pixel_size_y
            image[yoffset, xoffset] = mandel(real, imag, iters)
        
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)
    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()
