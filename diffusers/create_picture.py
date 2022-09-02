import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
 
MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = "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"
 
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
# pipe.to(DEVICE)
 
prompt = "a dog painted by Katsuhika Hokusai"
 
with autocast(DEVICE):
  image = pipe(prompt, guidance_scale=7.5)["sample"][0]
  image.save("output/test.png")