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

## Usage
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

As suggested in the paper, you can also discard the T5_xxl,
```python
# 12GB VRAM, clip_g + clip_l
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```

You can also enable cpu_offload to further resuce memory consumption,
```python
pipe.enable_sequential_cpu_offload()
```

