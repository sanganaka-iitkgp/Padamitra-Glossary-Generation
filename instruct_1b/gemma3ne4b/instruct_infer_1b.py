
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
import re

base_model_id="unsloth/gemma-3n-E4B-it"
adapter_path = "./unsloth-gemma-3ne4b-it-glossary-1b-best-model" 

print("Loading tokenizer and model...")
processor = AutoProcessor.from_pretrained(base_model_id)

processor.padding_side = "right"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,
    # llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=["prediction_coefs", "altup", "correction_coefs","lm_head"]
)
base_model = Gemma3nForConditionalGeneration.from_pretrained(
    base_model_id,
    device_map="auto",
    # device_map="cuda:0",
    quantization_config=bnb_config, 
    attn_implementation="eager"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()



print("Loading test dataset...")
test_dataset = load_from_disk("../data/pc_data_instruct_test_1b")
output_file = "gemma_3ne4b_final_test_results_detailed_1b.jsonl"
open(output_file, "w").close()

results = []

print("Running inference...")
for item in tqdm(test_dataset.select(range(2))):

    text=processor.apply_chat_template(
            item["prompt"],
            add_generation_prompt=True,
            tokenize=False
        )
    inputs = processor(text=text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=False
        )
 
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    response_str = processor.decode(generated_ids, skip_special_tokens=True)
    
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


print("Evaluation complete. Check jsonl")

