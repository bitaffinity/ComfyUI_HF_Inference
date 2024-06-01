from ..session import session

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
        response = session.post(endpoint, data=image)
        if response.status_code != 200:
            raise Exception(response.text)
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
        response = session.post(endpoint, data=image)
        if response.status_code != 200:
            raise Exception(response.text)
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
        response = session.post(endpoint, data=image)
        if response.status_code != 200:
            raise Exception(response.text)
        result = response.json()
        return {"ui": {"text": result}}

NODE_CLASS_MAPPINGS = {
    "Classification": Classification,
    "ObjectDetection": ObjectDetection,
    "Segmentation": Segmentation,
}