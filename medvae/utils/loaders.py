from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    Orientation,
    SpatialPad,
    ScaleIntensity,
    CropForeground,
    ScaleIntensityRange,
)
import torch
import torch.nn.functional as F
from monai.transforms import Transform
import torchvision
from PIL import Image

class MonaiNormalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return torchvision.transforms.Normalize(self.mean, self.std)(img)
    
class MonaiPad(Transform):
    def __init__(self, size, mode="constant", value=0):
        self.size = size
        self.mode = mode
        self.value = value

    def __call__(self, img):
        padding = [] 
        for i in range(len(img.shape) - 1, 0, -1):  # Start from the last dimension
            total_pad = max(self.size[i - 1] - img.shape[i], 0)
            padding.extend([total_pad // 2, total_pad - total_pad // 2])
        padded_img = F.pad(img, padding, mode=self.mode, value=self.value)
        return padded_img
    
class MonaiImageOpen(Transform):
    def __init__(self):
        pass
    
    def __call__(self, path):
        return Image.open(path)

"""
Custom transform to normalize and pad 2D images
@input: path to image (str)
@Output: padding (np.array)
"""
def load_2d(path: str, merge_channels: bool = False, dtype: torch.dtype = torch.float32, **kwargs):
    img_transforms = Compose(
        transforms=[
        MonaiImageOpen(),
        torchvision.transforms.ToTensor(),
        ScaleIntensity(channel_wise=True, minv=0, maxv=1),
        MonaiNormalize(mean=[0.5], std=[0.5]),],
        lazy=True,
    )
    
    try:
        img = img_transforms(path).as_tensor()
        
        if merge_channels:
            img = img.mean(0, keepdim=True)
        
        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384)), torch.zeros((1, 384, 384))


"""
Custom transform to normalize, crop, and pad 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""
def load_mri_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    mri_transforms = Compose(
        transforms=[LoadImage(),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensity(channel_wise=True, minv=0, maxv=1),
        MonaiNormalize(mean=[0.5], std=[0.5]),
        CropForeground(k_divisible=[16, 16, 16])],
        lazy=True,
    )
    
    try: 
        mr_augmented = mri_transforms(path).as_tensor()
        return mr_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 128, 128, 128))

"""
Custom transform to normalize, crop, and pad 3D CT volumes
@input: path to image (str)
@Output: padding (np.array)
"""
def load_ct_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    ct_transforms = Compose(
        transforms=[LoadImage(),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        MonaiNormalize(mean=[0.5], std=[0.5]),
        CropForeground(k_divisible=[16, 16, 16])],
        lazy=True,
    )
    
    try: 
        ct_augmented = ct_transforms(path).as_tensor()
        return ct_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 256, 256, 256))
