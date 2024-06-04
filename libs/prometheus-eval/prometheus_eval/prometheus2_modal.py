# Taken from @https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_inference.py
import os
import time
from pathlib import Path

import modal
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_prometheus2_config(
    yaml_path: Path = Path.cwd()
    / "libs"
    / "prometheus-eval"
    / "libs"
    / "prometheus-eval"
    / "prometheus_eval"
    / "configs.yaml",
) -> DictConfig | ListConfig:
    return OmegaConf.load(yaml_path)


cfg = get_prometheus2_config()


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.3",
        "torch==2.3.0",
        "transformers==4.41.2",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.23.2",
        "omegaconf==2.3.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .copy_local_file(
        "./libs/prometheus-eval/libs/prometheus-eval/prometheus_eval/configs.yaml",
        remote_path="/root/libs/prometheus-eval/libs/prometheus-eval/prometheus_eval/configs.yaml",
    )
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={"model_dir": cfg.vllm.model_dir, "model_name": cfg.vllm.model_name},
        secrets=[modal.Secret.from_name("hf-erland")],
    )
)

app = modal.App(cfg.client.app_name, image=image)

with image.imports():
    import vllm


@app.cls(
    secrets=[modal.Secret.from_name("hf-erland")],  # type: ignore
    **cfg.cls,
)
class Model:
    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        engine_args = AsyncEngineArgs(
            model=cfg.vllm.model_dir,
            tensor_parallel_size=int(cfg.cls.gpu.split(":")[-1]),
            **cfg.vllm.async_engine,
        )
        self.template = cfg.template

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method(keep_warm=cfg.generate.keep_warm)
    async def generate(self, user_question: str):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(**cfg.vllm.sampling_params)

        request_id = random_uuid()
        result_generator = self.engine.generate(
            user_question,
            sampling_params,
            request_id,
        )
        index = 0
        concat_string = ""
        async for output in result_generator:
            if output.outputs[0].text and "\ufffd" == output.outputs[0].text[-1]:
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            concat_string += text_delta

        return concat_string

    @modal.exit()
    def stop_engine(self):
        if int(cfg.cls.gpu.split(":")[-1]):
            import ray

            ray.shutdown()


@app.local_entrypoint()
def main():
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "What is the fable involving a fox and grapes?",
        "What were the major contributing factors to the fall of the Roman Empire?",
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "What is the product of 9 and 8?",
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    ]
    model = Model()
    for question in questions:
        print("Sending new request:", question, "\n\n")
        for text in model.generate.remote(question):
            print(text, end="", flush=text.endswith("\n"))
