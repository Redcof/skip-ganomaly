import pathlib
from xml.etree import ElementTree

import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class AspectResize(torch.nn.Module):
    """
   Resize image while keeping the aspect ratio.
   Extra parts will be covered with 255(white) color value
   """
    
    def __init__(self, size, background=255):
        super().__init__()
        from torchvision.transforms.transforms import _setup_size
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.background = background
    
    @staticmethod
    def fit_image_to_canvas(image: Image, canvas_width, canvas_height, background=255) -> Image:
        # Get the dimensions of the image
        image_width, image_height = image.size
        
        # Calculate the aspect ratio of the image
        image_aspect_ratio = image_width / float(image_height)
        
        # Calculate the aspect ratio of the canvas
        canvas_aspect_ratio = canvas_width / float(canvas_height)
        
        # Calculate the new dimensions of the image to fit the canvas
        if canvas_aspect_ratio > image_aspect_ratio:
            new_width = canvas_height * image_aspect_ratio
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = canvas_width / image_aspect_ratio
        
        # Resize the image to the new dimensions
        import PIL
        image = image.resize((int(new_width), int(new_height)), PIL.Image.BICUBIC)
        
        # Create a blank canvas of the specified size
        import numpy as np
        canvas = np.zeros((int(canvas_height), int(canvas_width), 3), dtype=np.uint8)
        canvas[:, :, :] = background
        
        # Calculate the position to paste the resized image on the canvas
        x = int((canvas_width - new_width) / 2)
        y = int((canvas_height - new_height) / 2)
        
        # Paste the resized image onto the canvas
        canvas[y:y + int(new_height), x:x + int(new_width)] = np.array(image)
        
        return PIL.Image.fromarray(canvas)
    
    def forward(self, image: Image) -> Image:
        image = self.fit_image_to_canvas(image, self.size[0], self.size[1], self.background)
        return image


class SIXraySD3AnomalyDataset(ImageFolder):
    """
    This dataset is a combination of Sixray easy and Smiths SD3 dataset. Below is the folder structure.
    
    sixray_sd3
        - train
            - 0.normal
                - 80% jpg negative(without threat) images form SD3
        - test
            - 1.abnormal
                - all jpg positive(with threat) images form SIXRay
            - 0.normal
                - 20% jpg negative(without threat) images form SD3
    """
    
    def __init__(self, data_dir, split='train', imsize=64, transform=None, target_transform=None):
        if transform is None:
            transform = transforms.Compose([
                AspectResize(imsize),
                transforms.ToTensor(),
                # transforms.RandomRotation(270),
                transforms.ColorJitter(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        super().__init__(os.path.join(data_dir, split), transform, target_transform)
