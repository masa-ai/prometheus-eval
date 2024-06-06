from typing import Any, Dict, List, Optional, Union

from loguru import logger

def dynamic_import(module_name):
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import {module_name}: {str(e)}")
        raise e

class VLLM:
    def __init__(
        self,
        name: str,
        num_gpus: int = 1,
        download_dir: Optional[str] = None,
        dtype: str = "auto",
        gpu_memory_utilization: int = 0.85,
        max_model_len: int = 8192,
        **kwargs,
    ) -> None:
        try:
            global LLM, SamplingParams
            vllm = dynamic_import("vllm")
            LLM = vllm.LLM
            SamplingParams = vllm.SamplingParams
        except Exception as e:
            raise ImportError(f"VLLM is not imported, to use `inference_engine` == 'vllm', you need to install vllm or install using the vllm setup")

        self.name: str = name

        self.model = LLM(
            model=self.name,
            tensor_parallel_size=num_gpus,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=download_dir,
            max_model_len=max_model_len,
            **kwargs,
        )

    def completions(
        self,
        prompts: List[str],
        use_tqdm: bool = False,
        **kwargs: Union[int, float, str],
    ) -> List[str]:
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(**kwargs)

        outputs = self.model.generate(prompts, params, use_tqdm=use_tqdm)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs

    def generate(
        self,
        prompts: List[str],
        use_tqdm: bool = False,
        **kwargs: Union[int, float, str],
    ) -> List[str]:
        params = SamplingParams(**kwargs)
        return self.model.generate(prompts, params, use_tqdm=use_tqdm)


class ModalVLLM:
    def __init__(self, app_name: str, tag: str) -> None:
        try:
            global modal
            modal = dynamic_import("modal")
        except Exception as e:
            raise ImportError(f"modal is not imported, to use `inference_engine` == 'modal', you need to install modal or install using the modal setup")

        self.app_name: str = app_name
        self.tag: str = tag
        self.modal_function = self.instantiate_modal_function(
            app_name=app_name, tag=tag
        )

    def instantiate_modal_function(
        self, app_name: str, tag: str
    ):
        try:
            modal_function = modal.Function.lookup(app_name=app_name, tag=tag)
        except modal.exception.NotFoundError as e:
            logger.warning(f"{tag} not found in {app_name}, return None. Error: {e}")
            modal_function = None
            raise ValueError(f"{tag} not found in Modal {app_name}")
        return modal_function

    def completions(
        self,
        prompts: List[str],
        use_tqdm: bool = False,
        **kwargs: Union[int, float, str],
    ) -> List[str]:
        return self.generate(prompts=prompts, use_tqdm=use_tqdm)

    def generate(
        self,
        prompts: List[str],
        use_tqdm: bool = False,
    ) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.modal_function.remote(prompt))
        return results


class OllamaVLLM:
    def __init__(self, name: str) -> None:
        try:
            global requests
            requests = dynamic_import("requests")
        except Exception as e:
            raise ImportError(f"requests is not imported, to use `inference_engine` == 'ollama', you need to install requests or install using the ollama setup")
        print("OllamaVLLM initialized")
        self.name = name

    def request_url(self, prompts: List[str]) -> List[Dict[str, Any]]:
        url = "http://localhost:11434/api/generate"
        results = []

        for prompt in prompts:
            data = {
                "model": self.name,
                "prompt": prompt,
                "raw": True,
                "stream": False,
            }

            response = requests.post(url, json=data)
            results.append(response.json())  # Assuming response is a JSON object

        return results

    def completions(self, prompts: List[str], **kwargs) -> List[str]:
        prompts = [prompt.strip() for prompt in prompts]

        outputs = self.request_url(prompts)
        outputs = [
            output["response"] for output in outputs
        ]  # Adjust based on actual response structure
        return outputs

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        return self.request_url(prompts)


class MockVLLM:
    def __init__(
        self,
    ) -> None:
        print("Mock VLLM initialized")

    def generate(
        self,
        prompts: List[str],
        use_tqdm: bool = False,
        **kwargs,
    ) -> List[str]:
        results = []
        for prompt in prompts:
            if "A or B" in prompt:
                results.append("Hello [RESULT] A")
            elif "1 and 5" in prompt:
                results.append("Hello [RESULT] 5")
            else:
                results.append("Response to: " + prompt)  # default fallback
        return results

    def completions(
        self,
        prompts: List[str],
        use_tqdm: bool = False,
        **kwargs,
    ) -> List[str]:
        # This mirrors the generate function in the mock
        return self.generate(prompts, use_tqdm, **kwargs)
