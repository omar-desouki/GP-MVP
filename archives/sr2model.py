# !pip install git+https://github.com/huggingface/diffusers.git
# https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages

import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch
import os


def super_resolve_image(
    image_source,
    model_id="CompVis/ldm-super-resolution-4x-openimages",
    output_path="ldm_generated_image.png",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    # load image from URL or local path
    if image_source.startswith("http://") or image_source.startswith("https://"):
        response = requests.get(image_source)
        low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # if not os.path.exists(image_source):
        #     raise FileNotFoundError(f"No such file or directory: '{image_source}'")
        low_res_img = Image.open(image_source).convert("RGB")

    # Store original dimensions
    original_size = low_res_img.size

    low_res_img = low_res_img.resize((128, 128))

    # run pipeline in inference (sample random noise and denoise)
    upscaled_image = pipeline(low_res_img, num_inference_steps=50, eta=1).images[0]

    # Resize upscaled image to original dimensions
    upscaled_image = upscaled_image.resize(original_size)

    # save image
    upscaled_image.save(output_path)


# # Example usage with URL
# url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
# super_resolve_image(url, output_path="test1.png")

# Example usage with local file path
file_path = "./input_imgs/man.png"
super_resolve_image(file_path, output_path="output_test.png")
