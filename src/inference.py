from typing import Optional

from modal import gpu, method

from .common import (
    MODEL_PATH,
    generate_prompt,
    output_vol,
    stub,
    VOL_MOUNT_PATH,
    user_model_path,
)


@stub.cls(
    gpu=gpu.A100(memory=40),
    network_file_systems={VOL_MOUNT_PATH: output_vol},
)
class OpenLlamaModel():
    def __init__(self, user: str, team_id: Optional[str] = None):
        import sys

        import torch
        from peft import PeftModel
        from transformers import LlamaForCausalLM, LlamaTokenizer

        self.user = user
        CHECKPOINT = user_model_path(self.user, team_id)

        load_8bit = False
        device = "cuda"

        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            model,
            CHECKPOINT,
            torch_dtype=torch.float16,
        )

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model
        self.device = device

    @method()
    def generate(
        self,
        input: str,
        max_new_tokens=128,
        **kwargs,
    ):
        import torch
        from transformers import GenerationConfig

        prompt = generate_prompt(self.user, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # print(tokens)
        generation_config = GenerationConfig(
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        return output.split("### Response:")[1].strip()


@stub.local_entrypoint()
def main(user: str):
    inputs = [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "What should we do next? Who should work on this?",
        "What are your political views?",
        "What did you work on yesterday?",
        "@here is anyone in the office?",
        "What did you think about the last season of Silicon Valley?",
        "Who are you?",
    ]
    model = OpenLlamaModel(user, None)
    for input in inputs:
        input = "U02ASG53F9S: " + input
        print(input)
        print(
            model.generate.remote(
                input,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                top_k=40,
                num_beams=1,
                max_new_tokens=600,
                repetition_penalty=1.2,
            )
        )
