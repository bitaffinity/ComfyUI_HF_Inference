import torch
from ..session import post

class Generation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "text": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "HF_Inference/Text/Generation"
    TITLE = "HF Text Generation"

    def inference(self, endpoint, text):
        json = {
            'inputs': text,
        }
        response = post(endpoint, json=json)
        result = response.json()
        generated = ''.join(x['generated_text'] for x in result)
        return generated 

class Translation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "text": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "HF_Inference/Text/Translation"
    TITLE = "HF Text Translation"

    def inference(self, endpoint, text):
        json = {
            'inputs': text,
        }
        response = post(endpoint, json=json)
        result = response.json()
        translation = ''.join(x['translation_text'] for x in result)
        return translation 

class QuestionAnswering:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "question": ("STRING", {"multiline": True}),
                "context": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "HF_Inference/Text/QuestionAnswering"
    TITLE = "HF Text Question Answering"

    def inference(self, endpoint, question, context):
        json = {
            'inputs': {
                'question': question,
                'context': context,
            },
        }
        response = post(endpoint, json=json)
        result = response.json()
        answer = result['answer']
        return answer

class FeatureExtraction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint": ("STRING", {}),
                "text": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "inference"
    CATEGORY = "HF_Inference/Text/FeatureExtraction"
    TITLE = "HF Text Feature Extraction"

    def inference(self, endpoint, text):
        json = {
            'inputs': text,
        }
        response = post(endpoint, json=json)
        result = response.json()
        cond = torch.tensor(result, dtype=torch.float16).to('cuda')
        return ([[cond, {}]],)

NODE_CLASS_MAPPINGS = {
    "FeatureExtraction": FeatureExtraction,
    "QuestionAnswering": QuestionAnswering,
    "Translation": Translation,
    "Generation": Generation,
}