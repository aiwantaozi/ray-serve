from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.gradio_integrations import GradioIngress

import gradio as gr

import asyncio
from transformers import pipeline

# 1. `pip install --upgrade huggingface_hub`
# 1. `pip install "ray[serve]" transformers requests torch`
# 2. `pip install gradio==3.50.2`
# 3. download the model `huggingface-cli download meta-llama/Llama-2-7b-chat-hf`
# 3. export `PYTORCH_ENABLE_MPS_FALLBACK=1` to use cpu
# 4. `serve run gradiobot:app`
# 5. access http://localhost:8000


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 9, "num_gpus": 0})
class TextGenerationModel:
    def __init__(self, model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer)

    def __call__(self, text):
        generated_list = self.generator(
            text, do_sample=True, min_length=20, max_length=200, truncation=True
        )
        generated = generated_list[0]["generated_text"]
        return generated


app_test_gen = TextGenerationModel.bind("meta-llama/Llama-2-7b-chat-hf")


# default is 1 cpu
@serve.deployment
class MyGradioServer(GradioIngress):
    def __init__(self, downstream_model: DeploymentHandle):
        self._d1 = downstream_model

        super().__init__(lambda: gr.Interface(self.fanout, "textbox", "textbox"))

    async def fanout(self, text):
        [result1] = await asyncio.gather(
            self._d1.remote(text)
        )
        return (
            f"[Generated text]\n{result1}\n\n"
        )


app = MyGradioServer.bind(app_test_gen)
