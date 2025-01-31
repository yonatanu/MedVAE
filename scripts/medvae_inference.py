import argparse, textwrap
from genericpath import isdir
from medvae.utils.extras import create_directory, cite_function, roi_size_calc
from medvae.utils.factory import create_model_and_transform
import torch
import nibabel as nib
import os
from tqdm import tqdm
import numpy as np
from os.path import join as pjoin
from monai.inferers import sliding_window_inference


def parse_arguments():
    parser = argparse.ArgumentParser(description='Use this to run inference with Med-VAE. This function is used when '
                                                 'you want to manually specify a folder containing an pretrained Med-VAE '
                                                 'model. ',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. These should contain files that you want the latent to be processed for'
                        'Remember *.pngs are for 2D images and *.nii.gz are for 3D images. ' 
                        'The filename should not have a "." in it apart from suffix')
    
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted latents will '
                             'have the same name as their source images.')
    parser.add_argument(
            '-model_name', type=str, required=True,
            help=(
                "There are six Med-VAE models that can be used for inference. Choose between:\n"
                "(1) medvae_4_1_2d: 2D images with a 4x compression in each dim (16x total) with a 1 channel latent.\n"
                "(2) medvae_4_3_2d: 2D images with a 4x compression in each dim (64x total) with a 3 channel latent.\n"
                "(3) medvae_8_1_2d: 2D images with an 8x compression in each dim (64x total) with a 1 channel latent.\n"
                "(4) medvae_8_4_2d: 2D images with an 8x compression in each dim (64x total) with a 4 channel latent.\n"
                "(5) medvae_4_1_3d: 3D images with a 4x compression in each dim (64x total) with a 1 channel latent.\n"
                "(6) medvae_8_1_3d: 3D images with an 8x compression in each dim (64x total) with a 1 channel latent.\n"
            )
        )
    parser.add_argument('-modality', type=str, required=True,
                        help='Modality of the input images. Choose between xray, ct, or mri.')
    parser.add_argument('-roi_size', type=int, default=160, required=False,
                        help='Region of interest size for 3D models. This is the maximum dimension size allowed for processing on the GPU.')
    
    parser.add_argument('-device', type=str, default='cuda', required=False,
                            help="Use this to set the device the inference should run with. Available options are 'cuda' "
                                "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                                "Use CUDA_VISIBLE_DEVICES=X medvae_inference [...] instead!")    
    
    # Print a message to cite the med-vae paper
    cite_function()
    
    args = parser.parse_args()
    
    # Check if input folder exists
    assert isdir(args.i), f"Input folder {args.i} does not exist."

    # Create output directory if it does not exist
    create_directory(args.o)    
    
    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    
    if args.device == 'cpu':
        # let's allow torch to use lots of threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help med-vae if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    
    return args, device

''' 
Inference function to take in a file path and model and return the latent
'''
def model_inference(fpath : str, model, transform, **kwargs):
    # Unpack the arguments
    model_name = kwargs.get('model_name')
    gpu_dim = kwargs.get('roi_size')
    device = kwargs.get('device')
    
    if '3d' in model_name:
        # Then just run inference on the patch
        def predict_latent(patch):
            with torch.no_grad():
                z, _, _ = model(patch, decode=False)
                return z
            
        img = transform(fpath).unsqueeze(0).to(device)
        
        roi_size = roi_size_calc(img.shape[-3:], target_gpu_dim=gpu_dim)
        rec = sliding_window_inference(inputs=img, roi_size=roi_size, sw_batch_size=1, mode="gaussian", predictor=predict_latent)
        latent = rec.squeeze().squeeze().detach().cpu().numpy()
    elif '2d' in model_name:
        img = transform(fpath, merge_channels='1_2d' in model_name).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, _, latent = model(img, decode=False)
        latent = latent.squeeze().squeeze().detach().cpu().numpy()

    return latent

def __init__():

    args, device = parse_arguments()
    
    # Build the model and transform
    model, transform = create_model_and_transform(args.model_name, args.modality, device)
    
    # Run inference on the input folder
    print("Running inference at {}".format(args.i))
    for fpath in tqdm(os.listdir(args.i), total = len(os.listdir(args.i))):
        latent = model_inference(pjoin(args.i, fpath), model, transform, **vars(args))
        
        # Save the latent
        nib.save(nib.Nifti1Image(latent, np.eye(4)), pjoin(args.o, fpath.split('.')[0] + ".nii.gz"))
    
    print("Inference complete! Output saved at {}".format(args.o))
    
def main():
    __init__()