from typing import AsyncIterator, Optional
import time
import modal

from .common import (
    MODEL_PATH,
    SYSTEM_PROMPT,
    VOL_MOUNT_PATH,
    app,
    output_vol,
    get_user_checkpoint_path,
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
            max_lora_rank=32,
            max_model_len=4096,
            # disable_custom_all_reduce=True,  # brittle as of v0.5.0
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def generate(
        self, input: list[dict], user: str, team_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        checkpoint_path = get_user_checkpoint_path(user, team_id)
        lora_request = LoRARequest(f"{user}-{team_id}", 1, checkpoint_path)

        tokenizer = await self.engine.get_tokenizer(lora_request=lora_request)

        prompt = tokenizer.apply_chat_template(
            conversation=input, tokenize=False, add_generation_prompt=True
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
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            tokens = len(request_output.outputs[0].token_ids)

        throughput = tokens / (time.time() - t0)
        print(f"🧠: Effective throughput of {throughput:.2f} tok/s")


@app.local_entrypoint()
def main(user: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.replace("{NAME}", "John Carmack")},
    ]
    inputs = [
        "Who are you?",
        "What should we do next? Who should work on this?",
        "What did you work on yesterday?",
        "@here is anyone in the office?",
        "What did you think about the last season of Silicon Valley?",
        "What were we just talking about?",
        "Summarize the conversation.",
    ]
    model = Inference()
    for input in inputs:
        print(input)
        messages.append({"role": "user", "content": "U02ASG53F9S: " + input})
        output_text = ""
        for output in model.generate.remote_gen(messages, user=user):
            print(output, end="")
            output_text += output
        print()
        messages.append({"role": "assistant", "content": output_text})
