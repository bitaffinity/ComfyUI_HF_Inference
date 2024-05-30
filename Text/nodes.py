import torch
from ..session import session

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
        response = session.post(endpoint, json=json)
        if response.status_code != 200:
            raise Exception(response.text)
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
        response = session.post(endpoint, json=json)
        if response.status_code != 200:
            raise Exception(response.text)
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
        response = session.post(endpoint, json=json)
        if response.status_code != 200:
            raise Exception(response.text)
        result = response.json()
        cond = torch.tensor(result, dtype=torch.float16).to('cuda')
        return ([[cond, {}]],)

NODE_CLASS_MAPPINGS = {
    "TextFeatureExtraction": FeatureExtraction,
    "QuestionAnswering": QuestionAnswering,
    "Translation": Translation,
}