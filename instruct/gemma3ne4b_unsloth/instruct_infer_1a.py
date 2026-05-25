import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
import time

# print("Sleeping......")
# time.sleep(2500)

base_model_id="unsloth/gemma-3n-E4B-it"
adapter_path = "./unsloth-gemma-3ne4b-it-glossary-1a-best-model" 
# adapter_path = "./unsloth-gemma-3ne4b-it-glossary-1a/checkpoint-9000" 

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
test_dataset = load_from_disk("../../data/data_instruct_test_1a")
output_file = "cleaned_results.jsonl"
# open(output_file, "w").close()

with open(output_file,"r") as file:
    lines=file.readlines()

broken=[61, 194, 490, 628, 1526, 1804, 2118, 2131, 2279, 2505, 2853, 2987, 2993, 3104]
# broken=[61]

results = []
print("Running inference...")
for idx,item in enumerate(tqdm(test_dataset)):
    if idx+1 not in broken:
        continue
     
    text=processor.apply_chat_template(
            item["prompt"],
            add_generation_prompt=True,
            tokenize=False
        )

    inputs = processor(text=text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            # do_sample=False,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            use_cache=False
        )
 
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    response_str = processor.decode(generated_ids, skip_special_tokens=True)
    
    expected_str = item["completion"][-1]["content"]
    try:
        expected_dict = json.loads(expected_str)["glossary"]
    except json.JSONDecodeError:
        expected_dict = expected_str
        
    clean_response_str = response_str.strip()
    if clean_response_str.startswith("```json"):
        clean_response_str = clean_response_str[7:]
    if clean_response_str.endswith("```"):
        clean_response_str = clean_response_str[:-3]
    clean_response_str = clean_response_str.strip()

    try:
        generated_dict = json.loads(clean_response_str)["glossary"]
    except json.JSONDecodeError:
        generated_dict = clean_response_str 
    
    step_result = {
        "prompt": item["prompt"][0]["content"],
        "expected_glossary": expected_dict,
        "generated_glossary": generated_dict,
        "raw_generated_text": response_str 
    }
    lines[idx]=json.dumps(step_result, ensure_ascii=False)+"\n"
    # with open(output_file, "a", encoding="utf-8") as f:
    #     f.write(json.dumps(step_result, ensure_ascii=False) + "\n")

with open(output_file, "w") as file:
    file.writelines(lines)

print("Evaluation complete. Check final_test_results_detailed_1a.json")