# Hugging Face hosted inference nodes for ComfyUI

Unofficial ComfyUI nodes for Hugging Face's inference API

Visit [the official docs](https://huggingface.co/docs/api-inference/detailed_parameters) for an overview of how the HF inference endpoints work

Find models by task on the [official website](https://huggingface.co/tasks)

## Installation

### Clone and install dependencies
```
git clone https://github.com/bitaffinity/ComfyUI_HF_Inference custom_nodes/ComfyUI_HF_Inference
cd custom_nodes/ComfyUI_HF_Inference
pip install -r requirements.txt
```

Export HF_AUTH_TOKEN with one of your [Hugging Face tokens](https://huggingface.co/settings/tokens)

### Run ComfyUI
`HF_AUTH_TOKEN=hf_1111111111111111111111111111111111 python main.py`

## Usage

### Text

* Feature Extraction (ie: T5 encoder embeddings, BERT, etc.)
* Question Answering (ie: roberta-base for QA)
* Translation (ie: T5 Small)
* Generation (ie: zephyr-7b-beta)

### Image

* Classification (ie: vit-base-patch16-224)
* Object Detection (ie: detr-resnet-50)
* Segmentation (ie: detr-resnet-50)