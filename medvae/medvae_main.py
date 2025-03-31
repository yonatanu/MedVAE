from medvae.utils.factory import create_model_and_transform
from medvae.utils.extras import roi_size_calc
import torch
from monai.inferers import sliding_window_inference

'''
Large Med-VAE class to abstract all the models.

This allows for interfacing with Med-VAE as a pytorch model.
Can be used for 2D and 3D inference / finetuning.

@param model_name: The name of the model to use. Choose between:
(1) medvae_4_1_2d: 2D images with a 4x compression in each dim (16x total) with a 1 channel latent.
(2) medvae_4_3_2d: 2D images with a 4x compression in each dim (64x total) with a 3 channel latent.
(3) medvae_8_1_2d: 2D images with an 8x compression in each dim (64x total) with a 1 channel latent.
(4) medvae_8_4_2d: 2D images with an 8x compression in each dim (64x total) with a 4 channel latent.
(5) medvae_4_1_3d: 3D images with a 4x compression in each dim (64x total) with a 1 channel latent.
(6) medvae_8_1_3d: 3D images with an 8x compression in each dim (64x total) with a 1 channel latent.

@param modality: Modality of the input images. Choose between xray, ct, or mri.

@param gpu_dim: The maximum dimension size allowed for processing on the GPU (default is 160).

@return (forward): The latent representation of the input image (torch.tensor).
'''
class MVAE(torch.nn.Module):
    def __init__(self, model_name : str, modality : str, gpu_dim=160):
        super(MVAE, self).__init__()
        
        self.model_name = model_name
        self.modality = modality
        
        self.model, self.transform = create_model_and_transform(self.model_name, self.modality)

        self.gpu_dim = gpu_dim
        
    def apply_transform(self, fpath: str):
        if '3d' in self.model_name:
            return self.transform(fpath).unsqueeze(0)
        elif '2d' in self.model_name:
            return self.transform(fpath, merge_channels='1_2d' in self.model_name).unsqueeze(0)
        else:
            raise ValueError(f"Model name {self.model_name} not supported. Needs to be a 2D or 3D model.")
        
    def get_transform(self):
        return self.transform
    
    def init_from_ckpt(self, ckpt_path: str, state_dict: bool = True):
        self.model.init_from_ckpt(ckpt_path, state_dict=state_dict)

    def forward(self, img: torch.tensor):
        
        if '3d' in self.model_name:
            
            # Then just run inference on the patch
            def predict_latent(patch):
                with torch.no_grad():
                    z, _, _ = self.model(patch, decode=False)
                    return z
            
            roi_size = roi_size_calc(img.shape[-3:], target_gpu_dim=self.gpu_dim)
            latent = sliding_window_inference(inputs=img, roi_size=roi_size, sw_batch_size=1, mode="gaussian", predictor=predict_latent)
            latent = latent.squeeze().squeeze()
            
        elif '2d' in self.model_name:
            
            with torch.no_grad():
                _, _, latent = self.model(img, decode=False)
                
            latent = latent.squeeze().squeeze()
        
        return latent