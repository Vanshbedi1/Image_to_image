import torch

from diffusers import (
    StableDiffusionXLPipeline,
)

from PIL import Image


class ReferenceImageGenerator:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True
        )

        self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()


    def generate(
        self,
        prompt,
        reference_images=None,
        negative_prompt="",
        width=1024,
        height=1024,
        steps=30,
        guidance=7.0,
        seed=42
    ):

        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        )

        return result.images[0]