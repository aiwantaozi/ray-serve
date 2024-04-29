from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
import torch

from ray import serve
from ray.serve.handle import DeploymentHandle

# 1. `pip install "ray[serve]" requests torch diffusers==0.12.1 transformers`
# 2. `serve run stable_diffusion:entrypoint`

app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt)
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")


@serve.deployment()
class StableDiffusionV2:
    def __init__(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5")
        self.pipe = self.pipe.to("mps")
        self.pipe.enable_attention_slicing()

    def generate(self, prompt: str):
        assert len(prompt), "prompt parameter cannot be empty"

        image = self.pipe(prompt).images[0]
        return image


entrypoint = APIIngress.bind(StableDiffusionV2.bind())
