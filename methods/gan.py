from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import os

from torchvision.utils import save_image
from PIL import Image

models = { 'CUPHEAD': 'cuphead.pth', 'STARRY NIGHT': 'starry_night.pth', 'MOSAIC': 'mosaic.pth' }

def apply_style(file_name, style) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(f'models/{models[style]}', map_location = torch.device(device)))
    transformer.eval()

    transform = input_ransform(image_size=640)
    # Prepare input
    image_tensor = Variable(transform(Image.open(file_name))).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    # Save image
    fn = file_name.split("/")[-1]
    output_path = f"outputs/{style}-{fn}"
    save_image(stylized_image, output_path)

    return output_path
