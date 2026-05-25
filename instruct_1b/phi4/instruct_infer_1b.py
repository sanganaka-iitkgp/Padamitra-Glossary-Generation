import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import re

base_model_id="unsloth/phi-4"
adapter_path = "./phi4-glossary-1b-best-model" 

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Loading model with vLLM...")
llm = LLM(
    model=base_model_id,
    enable_lora=True,
    max_lora_rank=256,       
    quantization="bitsandbytes",
    load_format="bitsandbytes", 
    dtype="bfloat16",
    gpu_memory_utilization=0.50,
    tensor_parallel_size=1,  
    max_model_len=4096,
)

lora_request = LoRARequest(
    lora_name="glossary-adapter",
    lora_int_id=1,
    lora_path=adapter_path,
)

sampling_params = SamplingParams(
    temperature=0.0,  
    max_tokens=2048,
)

print("Loading test dataset...")
test_dataset = load_from_disk("../../jumbled_data/jumbled_pc_data_instruct_test_1b")
# test_dataset=test_dataset.select(range(2))

output_file = "jumbled_results_1b.jsonl"
open(output_file, "w").close()


print("Preparing prompts...")
prompts = [
    tokenizer.apply_chat_template(
        item["prompt"],
        add_generation_prompt=True,
        tokenize=False, 
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


print("Evaluation complete. Check final_test_results_detailed_1b.json")