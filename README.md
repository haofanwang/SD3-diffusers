# SD3-diffusers
A simple usage of Stable Diffusion 3 in diffusers.

<div align="center">
<img src='/image.jpg' width = 900 >
</div>

## Environment
```
pip install -U diffusers
pip install -U safetensors
pip install -U transformers
pip install torch>=2.1.1
```

## Text-To-Image
```python
import torch
from diffusers import StableDiffusion3Pipeline

# 24GB VRAM, clip_g + clip_l + T5_xxl
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# generate in 1024*1024
image = pipe(
    "A cat holding a sign that says Hello SD3",
    negative_prompt="",
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]

# save
image.save("image.jpg")
```

You can load the T5-XXL model in 8 bits using the bitsandbytes library to reduce the memory requirements further.
```python
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig

# Make sure you have `bitsandbytes` installed. 
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
)
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=text_encoder,
    device_map="balanced",
    torch_dtype=torch.float16
)
```

As suggested in the paper, you can drop the T5_xxl,
```python
# 12GB VRAM, clip_g + clip_l
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```

You can also enable cpu_offload to further reduce memory consumption,
```python
# less than 8GB VRAM
pipe.enable_sequential_cpu_offload()
```

## Image-To-Image
```python
import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipe(prompt, image=init_image).images[0]
```

## LoRA
```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to("cuda")

pipe.load_lora_weights("nerijs/pixel-art-medium-128-v0.1", weight_name="pixel-art-medium-128-v0.1.safetensors")

image = pipe(
    "A cat, pixel art style",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=5.0,
).images[0]
```
Refer to [here](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sd3.md) for training details.

## Acknowledgement
Thanks to the Stability AI team for making Stable Diffusion 3 happen and [HuggingFace Team](https://huggingface.co/blog/sd3).
