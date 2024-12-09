from typing import AsyncIterator, Optional
import time
import modal

from .common import (
    MODEL_PATH,
    SYSTEM_PROMPT,
    VOL_MOUNT_PATH,
    app,
    output_vol,
    user_model_path,
)


vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.3post1", "fastapi[standard]==0.115.4"
)

with vllm_image.imports():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid

    from vllm.lora.request import LoRARequest


@app.cls(
    image=vllm_image,
    gpu="H100",
    container_idle_timeout=10 * 60,
    timeout=5 * 60,
    allow_concurrent_inputs=100,
    volumes={VOL_MOUNT_PATH: output_vol},
)
class Inference:
    @modal.enter()
    def enter(self):
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=1,
            enable_lora=True,
            # disable_custom_all_reduce=True,  # brittle as of v0.5.0
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def generate(
        self, input: str, user: str, team_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        checkpoint_path = user_model_path(user, team_id) / "epoch_0"
        lora_request = LoRARequest(f"{user}-{team_id}", 1, checkpoint_path)

        tokenizer = await self.engine.get_tokenizer(lora_request=lora_request)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input},
        ]
        prompt = tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=False
        )

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if (
                request_output.outputs[0].text
                and "\ufffd" == request_output.outputs[0].text[-1]
            ):
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"🧠: Effective throughput of {throughput:.2f} tok/s")


@app.local_entrypoint()
def main(user: str):
    inputs = [
        "Tell me about alpacas.",
        "What should we do next? Who should work on this?",
        "What are your political views?",
        "What did you work on yesterday?",
        "@here is anyone in the office?",
        "What did you think about the last season of Silicon Valley?",
        "Who are you?",
    ]
    model = Inference()
    for input in inputs:
        input = "U02ASG53F9S: " + input
        print(input)
        for output in model.generate.remote_gen(input, user=user):
            print(output, end="")
        print()
