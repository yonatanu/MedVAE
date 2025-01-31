'''
Miscellaneous utility functions.
'''

import os


def create_directory(directory: str) -> None:
    '''
    Create a directory if it does not exist.

    Parameters:
        directory (str): The directory to create.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def cite_function():
    '''
    Print a message to cite the Med-VAE paper.
    '''
    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using Med-VAE: TODO \n#######################################################################\n")


''' 
Calculate the region of interest (ROI) size for each dimension of the input image shape.

This function determines the appropriate ROI size based on the target GPU dimension.
If a dimension exceeds the target GPU dimension, it finds the largest power of 2 that 
results in a size less than the target dimension.

@param image_shape: A tuple or list representing the shape of the input image (e.g., (depth, height, width)).
@param target_gpu_dim: The maximum dimension size allowed for processing on the GPU (default is 160).
@return: A list of calculated ROI sizes for each dimension of the input image.
'''
def roi_size_calc(image_shape, target_gpu_dim=160):
    roi_size = []
    for dim in image_shape:   
        if dim > target_gpu_dim:
            target_shape = target_gpu_dim
            # For loop for powers of 2
            for power in [2**i for i in range(8)]:
                if dim // power < target_gpu_dim:
                    roi_size.append(dim // power)
                    break
        else:
            roi_size.append(dim)
    
    return roi_size