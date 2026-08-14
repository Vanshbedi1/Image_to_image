# image_generator.py

import torch
from diffusers import StableDiffusionXLPipeline


class LocalImageGenerator:

    def __init__(self, model_id):

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

        self.pipe = (
            StableDiffusionXLPipeline
            .from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
        )

        self.pipe.to(self.device)


    def generate(
        self,
        prompt,
        negative_prompt="",
        width=1024,
        height=1024,
        steps=40,
        guidance=7,
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