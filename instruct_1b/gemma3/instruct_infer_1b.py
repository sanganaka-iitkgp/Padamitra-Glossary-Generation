import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import re 
import time
import torch


def wait_for_vram(target_gb=25, device=0, check_interval=10):
    target_bytes = target_gb * (1024 ** 3) 
    print(f"Waiting for {target_gb}GB of free VRAM to become available...")
    
    while True:
        free_vram, total_vram = torch.cuda.mem_get_info(device)
        
        if free_vram >= target_bytes:
            print(f"\nSuccess! {free_vram / (1024**3):.2f}GB free. Firing up the model!")
            break
            
        print(f"Current free VRAM: {free_vram / (1024**3):.2f}GB. Retrying in {check_interval}s...", end='\r')
        time.sleep(check_interval)



def main():
    # wait_for_vram(target_gb=40, device=0, check_interval=10)

    base_model_id="google/gemma-3-12b-it"
    adapter_path = "./gemma3-it-glossary-1b-best-model" 

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print("Loading model with vLLM...")
    llm = LLM(
        model=base_model_id,
        enable_lora=True,
        max_lora_rank=128,           # match the rank used during your LoRA training
        quantization="bitsandbytes",
        load_format="bitsandbytes", 
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        tensor_parallel_size=1,     # set to 2 if using 2 GPUs
        max_model_len=4096,
    )

    lora_request = LoRARequest(
        lora_name="glossary-adapter",
        lora_int_id=1,
        lora_path=adapter_path,
    )

    sampling_params = SamplingParams(
        temperature=0.0,  
        max_tokens=1024,
    )

    print("Loading test dataset...")
    test_dataset = load_from_disk("../../jumbled_data/jumbled_pc_data_instruct_test_1b")
    # test_dataset=test_dataset.select(range(2))

    output_file = "gemma3_jumbled_results_1b.jsonl"
    open(output_file, "w").close()



    print("Preparing prompts...")
    prompts = [
        tokenizer.apply_chat_template(
            item["prompt"],
            add_generation_prompt=True,
            tokenize=False,          # vLLM takes raw strings, not tensors
        )
        for item in tqdm(test_dataset, desc="Templating")
    ]

    print("Running batched inference...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)


    print("Writing results...")
    for item, output in tqdm(zip(test_dataset, outputs), total=len(test_dataset)):
        response_str = output.outputs[0].text

        expected_str = item["completion"][-1]["content"]

        
        expected_reasoning_str=expected_str.split("### FINAL GENERATION:")[0].strip().strip("\n")
        pattern = r'\[[^\]]*\]'
        expected_padacheda_list=re.findall(pattern, expected_reasoning_str)

        expected_glossary_str=expected_str.split("### FINAL GENERATION:")[-1].strip().strip("\n")
        
        try:
            if expected_glossary_str.startswith("```json"):
                expected_glossary_str = expected_glossary_str[7:]
            if expected_glossary_str.endswith("```"):
                expected_glossary_str = expected_glossary_str[:-3]
            expected_dict = json.loads(expected_glossary_str)["glossary"]
            
        except json.JSONDecodeError:
            expected_dict = expected_glossary_str


        generated_reasoning_str=response_str.split("### FINAL GENERATION:")[0].strip().strip("\n")
        pattern = r'\[[^\]]*\]'
        generated_padacheda_list=re.findall(pattern, generated_reasoning_str)

        generated_glossary_str=response_str.split("### FINAL GENERATION:")[-1].strip().strip("\n")

        try:
            if generated_glossary_str.startswith("```json"):
                generated_glossary_str = generated_glossary_str[7:]
            if generated_glossary_str.endswith("```"):
                generated_glossary_str = generated_glossary_str[:-3]
            generated_dict = json.loads(generated_glossary_str)["glossary"]
        except json.JSONDecodeError:
            generated_dict = generated_glossary_str 
        
        step_result = {
            "prompt": item["prompt"][0]["content"],
            "expected_padaccheda" : expected_padacheda_list,
            "expected_glossary": expected_dict,
            "generated_padaccheda":generated_padacheda_list,
            "generated_glossary": generated_dict,
            "raw_generated_text": response_str 
        }

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_result, ensure_ascii=False) + "\n")


if __name__=="__main__":
    main()

    
print("Evaluation complete. Results saved......")
