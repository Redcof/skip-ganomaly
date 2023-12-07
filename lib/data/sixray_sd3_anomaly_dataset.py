import pathlib
from xml.etree import ElementTree

import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
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


class SIXraySD3AnomalyDataset(data.Dataset):
    """
    This dataset is a combination of Sixray easy and Smiths SD3 dataset. Below is the folder structure.
    
    sixray_sd3
        - train
            - normal
                - 80% jpg negative(without threat) images form SD3
        - test
            - abnormal
                - all jpg positive(with threat) images form SIXRay
            - normal
                - 20% jpg negative(without threat) images form SD3
    """
    
    def __init__(self, data_dir, split='train',
                 imsize=64, transform=None, target_transform=None, float_precision=32):
        assert float_precision in (32, 64), "Required 32 or 64 but {} is given".format(float_precision)
        assert split in ('train', 'test'), "Required 'train' or 'test' but {} is given".format(split)
        if float_precision == 32:
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64
        sixray_sd3 = pathlib.Path(data_dir)
        normal_img_path = sixray_sd3 / split / "normal"
        abnormal_img_path = sixray_sd3 / split / "abnormal"
        assert os.path.exists(normal_img_path), "The path: '%s' does not exists" % normal_img_path
        if split == "test":
            assert os.path.exists(abnormal_img_path), "The path: '%s' does not exists" % abnormal_img_path
        self.float_precision = float_precision
        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        self.split = split
        # load normal filenames
        self.filenames = [os.path.join(normal_img_path, file) for file in os.listdir(normal_img_path)]
        if split == "test":
            # load abnormal filenames
            self.filenames.extend([os.path.join(abnormal_img_path, file) for file in os.listdir(abnormal_img_path)])
    
    def get_img(self, img_path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.Compose([
                AspectResize(self.imsize),
                transforms.ToTensor(),
                # transforms.RandomRotation(270),
                transforms.ColorJitter(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img)
        return img.type(self.dtype)
    
    @staticmethod
    def read_pascal_voc(xml_file) -> []:
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        
        list_with_all_boxes = []
        
        for boxes in root.iter('object'):
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            
            class_name = boxes.find("./name").text
            
            list_with_single_boxes = (xmin, ymin, xmax, ymax)
            list_with_all_boxes.append((class_name, list_with_single_boxes))
        
        return list_with_all_boxes
    
    def __getitem__(self, index):
        filepath = self.filenames[index]
        img = self.get_img(filepath)
        return img
    
    def __len__(self):
        return len(self.filenames)
