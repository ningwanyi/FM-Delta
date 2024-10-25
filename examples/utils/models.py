from diffusers import DiffusionPipeline
from transformers import GPT2Model, BertModel, ResNetForImageClassification, AutoModelForCausalLM
import torch


def read_models(base_name, finetuned_name, dtype=torch.float32):
    if "stable-diffusion" in base_name.lower():
        base_model = DiffusionPipeline.from_pretrained(base_name, torch_dtype=dtype).unet
        finetuned_model = DiffusionPipeline.from_pretrained(finetuned_name, torch_dtype=dtype).unet
    elif "bert" in base_name.lower():
        base_model = BertModel.from_pretrained(base_name, torch_dtype=dtype)
        finetuned_model = BertModel.from_pretrained(finetuned_name, torch_dtype=dtype)
    elif "resnet50" in base_name.lower():
        base_model = ResNetForImageClassification.from_pretrained(base_name, torch_dtype=dtype)
        finetuned_model = ResNetForImageClassification.from_pretrained(finetuned_name, torch_dtype=dtype)
    elif "gpt2" in base_name.lower():
        base_model = GPT2Model.from_pretrained(base_name, torch_dtype=dtype)
        finetuned_model = GPT2Model.from_pretrained(finetuned_name, torch_dtype=dtype)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_name, torch_dtype=dtype)
    return base_model, finetuned_model


def read_single_model(name, dtype=torch.float32):
    if "stable-diffusion" in name.lower():
        model = DiffusionPipeline.from_pretrained(name, torch_dtype=dtype).unet
    elif "bert" in name.lower():
        model = BertModel.from_pretrained(name, torch_dtype=dtype)
    elif "resnet50" in name.lower():
        model = ResNetForImageClassification.from_pretrained(name, torch_dtype=dtype)
    elif "gpt2" in name.lower():
        model = GPT2Model.from_pretrained(name, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model

