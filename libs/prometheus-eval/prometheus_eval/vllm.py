from typing import Any, Dict, List, Optional, Union

import requests
from vllm import LLM, SamplingParams


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
        self.name: str = name

        self.model: LLM = LLM(
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


class OllamaVLLM:
    def __init__(self, name: str) -> None:
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

    def generate(self, prompts: List[str], **kwargs):
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
