from ..session import post
from io import BytesIO
from PIL import Image
import numpy as np
import torch

class TextToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "inference"
    CATEGORY = "HF_Inference/Image"
    TITLE = "HF Image TextToImage"

    def inference(self, endpoint, prompt):
        payload = {"inputs": prompt}
        response = post(endpoint, json=payload)
        
        result = BytesIO(response.content)
        image = Image.open(result)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        mask = torch.zeros((64, 64), device="cpu")
        return (image, mask)

class Classification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "HF_Inference/Image"
    TITLE = "HF Image Classification"

    def inference(self, endpoint, image):
        response = post(endpoint, data=image)
        result = response.json()
        return result
    
class ObjectDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "HF_Inference/Image"
    TITLE = "HF Image Object Detection"

    def inference(self, endpoint, image):
        response = post(endpoint, data=image)
        result = response.json()
        return {"ui": {"text": result}}
from base64 import b64decode

class Segmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "images": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "inference"
    OUTPUT_NODE = True
    CATEGORY = "HF_Inference/Image"
    TITLE = "HF Image Segmentation"

    def inference(self, endpoint, images):
        for image in images:
            image_bytes = BytesIO()
            pil_img = Image.fromarray(
                np.clip(
                    image.cpu().numpy() * 255.0,
                    0,
                    255,
                ).astype(np.uint8)
            )
            pil_img.save(image_bytes, format='png')

            image_bytes.seek(0)
            image_bytes = image_bytes.read()
            print(len(image_bytes))
            
            response = post(endpoint, data=image_bytes)
            result = response.json()

            for item in result:
                label = item['label']
                mask_data = item['mask']

                mask_img = Image.open(BytesIO(b64decode(mask_data)))

                pil_img.paste(mask_img, (0, 0), mask_img)            
            pil_img = np.array(pil_img).astype(np.float32) / 255.0
            pil_img = torch.from_numpy(pil_img)[None,]
        return {"ui": {"images": [pil_img, ]}}

NODE_CLASS_MAPPINGS = {
    "Classification": Classification,
    "ObjectDetection": ObjectDetection,
    "Segmentation": Segmentation,
    "TextToImage": TextToImage,
}