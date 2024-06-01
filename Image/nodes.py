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
    CATEGORY = "HF_Inference/Image/TextToImage"
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
    CATEGORY = "HF_Inference/Image/Classification"
    TITLE = "HF Image Classification"

    def inference(self, endpoint, image):
        response = post(endpoint, data=image)
        result = response.json()
        return {"ui": {"text": result}}
    
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
    CATEGORY = "HF_Inference/Image/ObjectDetection"
    TITLE = "HF Image Object Detection"

    def inference(self, endpoint, image):
        response = post(endpoint, data=image)
        result = response.json()
        return {"ui": {"text": result}}
    
class Segmentation:
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
    CATEGORY = "HF_Inference/Image/Segmentation"
    TITLE = "HF Image Segmentation"

    def inference(self, endpoint, image):
        response = post(endpoint, data=image)
        result = response.json()
        return {"ui": {"text": result}}

NODE_CLASS_MAPPINGS = {
    "Classification": Classification,
    "ObjectDetection": ObjectDetection,
    "Segmentation": Segmentation,
    "TextToImage": TextToImage,
}