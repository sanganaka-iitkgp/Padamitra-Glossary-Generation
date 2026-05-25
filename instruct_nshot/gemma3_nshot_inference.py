import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from huggingface_hub import login
import re
import ast
from tqdm.auto import tqdm
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


wait_for_vram(target_gb=30, device=0, check_interval=10)



login()

model_id = "google/gemma-3-12b-it"  # Or "Qwen/Qwen3.5-9B"
model_name="gemma3"

# N_SHOTS = [0,1,5,10]
# N_SHOTS = [5,10]
N_SHOTS = [20]

def filter_sans_to_english(dataset):
    filtered_dataset=dataset.filter(lambda example: example['source_language']=='sanskrit' and example['target_language']=='english')
    return filtered_dataset

def fix_malformed_glossary(broken_dict):
    full_str = ", ".join([f"{k}: {v}" for k, v in broken_dict.items()])
    
    parts = full_str.split(":")
    
    if len(parts) <= 1:
        return broken_dict
        
    fixed_dict = {}
    current_key = parts[0].strip()
    
    for i in range(1, len(parts)):
        chunk = parts[i]
        
        if i == len(parts) - 1:
            fixed_dict[current_key] = chunk.strip()
        else:

            last_comma_idx = chunk.rfind(",")
            
            if last_comma_idx == -1:
                last_space_idx = chunk.rfind(" ")
                val = chunk[:last_space_idx].strip()
                next_key = chunk[last_space_idx:].strip()
            else:
                val = chunk[:last_comma_idx].strip()
                next_key = chunk[last_comma_idx+1:].strip()
                
            fixed_dict[current_key] = val
            current_key = next_key
            
    return fixed_dict

def parse_dirty_glossary(dirty_str):
    dirty_str = str(dirty_str).strip()
    
    try:
        clean_str = re.sub(r',\s*\}', '}', dirty_str)
        parsed = ast.literal_eval(clean_str)
        if isinstance(parsed, dict):
            return {str(k).strip(): str(v).strip() for k, v in parsed.items()}
    except Exception:
        pass

    clean_str = re.sub(r'\},?\n?\}?$', '', dirty_str)
    clean_str = clean_str.strip('{ \n\r')
    
    glossary_dict = {}
    pairs = clean_str.split(',\n')
    
    for pair in pairs:
        if ':' in pair:
            key, val = pair.split(':', 1)
            key = key.strip(' "\'')
            val = val.strip(' "\'')
            glossary_dict[key] = val
            
    return fix_malformed_glossary(glossary_dict)

def safe_parse(s):
    s = s.strip()

    # remove code block markers
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
    if s.endswith("```"):
        s = s[:-3]

    s = s.strip()

    # try JSON first
    try:
        return json.loads(s)
    except:
        pass

    # fallback: Python dict
    try:
        return ast.literal_eval(s)
    except:
        return s

def build_n_shot_messages(target_shloka, target_trans, train_dataset, n_shots=3):
    messages_1a,messages_1b = [], []
    system_prompt_1a = (
        "You are a Sanskrit-to-English glossary extraction engine. "
        "Given a Sanskrit shloka and its English translation, output a single JSON object mapping each Sanskrit word to its English meaning.\n\n"
        "Rules:\n"
        "1. Output ONLY a Python dict. Start with '{', end with '}'. No other text.\n"
        "2. Keys are Sanskrit words (or sandhi-resolved tokens) as they appear in the shloka.\n"
        "3. Values are their English meanings as evidenced by the translation.\n"
        "4. No markdown, no code fences, no explanation, no preamble.\n\n"
        "Format: {'glossary': {'sanskrit_word': 'english_meaning', ...}}"
    )

    system_prompt_1b = (
        "You are a Sanskrit scholar and glossary extraction engine. "
        "Given a Sanskrit shloka and its English translation, follow these two steps exactly.\n\n"

        "## STEP 1 — REASONING:\n"
        "Perform Padaccheda (resolve sandhi, keep samasa compound words intact). "
        "Analyze the shloka word by word using the translation as a reference. "
        "End your reasoning with the resolved word sequence as: [word1, word2, ...].\n\n"

        "## STEP 2 — FINAL GENERATION:\n"
        "Output the glossary as a code block. Rules for the output:\n"
        "1. Output exactly ONE valid Python dict with a single key 'glossary'.\n"
        "2. All keys are the Padaccheda-resolved Sanskrit words, in the order they appear in the shloka.\n"
        "3. All values are their English meanings, derived strictly from the provided translation.\n"
        "4. No trailing commas.\n"
        "5. No extra keys, no nested objects — flat dictionary only.\n\n"

        "## RESPONSE FORMAT (follow exactly, do not deviate):\n"
        "### REASONING:\n"
        "<your padaccheda analysis ending with the resolved word list>\n\n"
        "### FINAL GENERATION:\n"
        "{'glossary': {'sanskrit_word': 'english_meaning', ...}}\n"

        "Do NOT include any text outside these two sections. "
        "Do NOT add greetings, notes, or explanations after the output block."
    )
    messages_1a.append({"role": "system", "content": system_prompt_1a})
    messages_1b.append({"role": "system", "content": system_prompt_1b})
    for i in range(n_shots):
        ex = train_dataset[i]
        
        shlok = ex['shlok']
        translation = ex['translation']
        clean_glossary = parse_dirty_glossary(ex['glossary'])
        
        resolved_sequence = list(clean_glossary.keys())
        if not resolved_sequence:
            print("Oh my habibiii")
        
        prompt_1a = f"Generate a Sanskrit-English glossary mapping for the following shloka based on its translation.\n\nShloka:\n{shlok}\n\nTranslation:\n{translation}"
        assistant_response_1a = f'{{"glossary": {clean_glossary}}}'


        
        sequence_string = ", ".join(resolved_sequence)
        prompt_1b = f"### INPUT:\nYou are an expert in Sanskrit. Perform Padaccheda (word resolution) on the following Sanskrit shloka, then generate the glossary mapping using the generated Padaccheda and provided English translation.\n\nShloka:\n{shlok}\n\nTranslation:\n{translation}"
        reasoning_text = (
            "To build the glossary, I first need to divide the shloka into its semantic parts. "
            "By analyzing the given shloka and referencing the translation, I will break down the sandhi "
            "while keeping samasa (compound words) intact. "
            f"Upon dividing this, the resolved sequence of words is: [{sequence_string}]."
        )
        
        glossary_json = json.dumps({"glossary": clean_glossary}, ensure_ascii=False)
        output_1b = f"### REASONING:\n{reasoning_text}\n\n### FINAL GENERATION:\n```json\n{glossary_json}\n```"
        assistant_response_1b = output_1b

        
        messages_1a.append({"role": "user", "content": prompt_1a})
        messages_1a.append({"role": "assistant", "content": assistant_response_1a})

        messages_1b.append({"role": "user", "content": prompt_1b})
        messages_1b.append({"role": "assistant", "content": assistant_response_1b})

        
    target_prompt_1a = f"Generate a Sanskrit-English glossary mapping for the following shloka based on its translation.\n\nShloka:\n{target_shloka}\n\nTranslation:\n{target_trans}"

    target_prompt_1b = f"### INPUT:\nYou are an expert in Sanskrit. Perform Padaccheda (word resolution) on the following Sanskrit shloka, then generate the glossary mapping using the generated Padaccheda and provided English translation.\n\nShloka:\n{target_shloka}\n\nTranslation:\n{target_trans}"
    
    messages_1a.append({"role": "user", "content": target_prompt_1a})
    messages_1b.append({"role": "user", "content": target_prompt_1b})

    return messages_1a, messages_1b

if __name__ == "__main__":
    
    print(f"Loading vLLM engine for {model_id}...")

    llm = LLM(
        model=model_id, 
        tensor_parallel_size=1,   ######### increase when multi gpu avail 
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="bfloat16"
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=1200,
    )

    dataset = load_dataset("sanganaka/padamitra")
    train_dataset = filter_sans_to_english(dataset["train"])
    test_dataset = filter_sans_to_english(dataset["test"])


    for shot in N_SHOTS:
        print(f"Starting for {shot} shot inference....")
        output_file_1a = f"1a/{model_name}/{model_name}_vllm_{shot}_shot_results.jsonl"
        output_file_1b = f"1b/{model_name}/{model_name}_vllm_{shot}_shot_results.jsonl"
        open(output_file_1a, "w").close()
        open(output_file_1b, "w").close()

        print("Building and formatting all prompts...")
        all_formatted_prompts_1a = []
        all_formatted_prompts_1b = []
        
        # for item in test_dataset.select(range(20)):
        for item in test_dataset:
            messages_1a, messages_1b = build_n_shot_messages(item["shlok"], item["translation"], train_dataset, n_shots=shot)
            
            prompt_string_1a = tokenizer.apply_chat_template(
                messages_1a,
                add_generation_prompt=True,
                tokenize=False 
            )
            prompt_string_1b = tokenizer.apply_chat_template(
                messages_1b,
                add_generation_prompt=True,
                tokenize=False 
            )
            all_formatted_prompts_1a.append(prompt_string_1a)
            all_formatted_prompts_1b.append(prompt_string_1b)
        #     print(f"============={shot}shot 1a============")
        #     print(prompt_string_1a)
        #     print(f"============={shot}shot 1b============")
        #     print(prompt_string_1b)
        #     break
        # continue
        print(f"Running vLLM inference on {len(all_formatted_prompts_1a)} prompts...")
        
        outputs_1a = llm.generate(all_formatted_prompts_1a, sampling_params)
        outputs_1b = llm.generate(all_formatted_prompts_1b, sampling_params)

        print("Parsing and saving results of 1a...")
        for i, output in enumerate(tqdm(outputs_1a)):
            item = test_dataset[i]
            target_shloka = item["shlok"]
            
            response_str = output.outputs[0].text
            
            clean_response_str = response_str.strip()
            if clean_response_str.startswith("```json"):
                clean_response_str = clean_response_str[7:]
            if clean_response_str.endswith("```"):
                clean_response_str = clean_response_str[:-3]
            clean_response_str = clean_response_str.strip()

            try:
                generated_dict = json.loads(clean_response_str)
                if "glossary" in generated_dict:
                    generated_dict = generated_dict["glossary"]
                    
            except json.JSONDecodeError:
                try:
                    generated_dict = ast.literal_eval(clean_response_str)
                    if "glossary" in generated_dict:
                        generated_dict = generated_dict["glossary"]
                except (ValueError, SyntaxError):
                    generated_dict = clean_response_str
                
            step_result = {
                "shloka": target_shloka,
                "translation":item['translation'],
                "expected_glossary": parse_dirty_glossary(item["glossary"]),
                "generated_glossary": generated_dict,
                "raw_text": response_str
            }
            
            with open(output_file_1a, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_result, ensure_ascii=False) + "\n")

        print("Parsing and saving results of 1b...")
        for i, output in enumerate(tqdm(outputs_1b)):
            item = test_dataset[i]
            target_shloka = item["shlok"]
            
            response_str = output.outputs[0].text
            
            clean_response_str = response_str.strip()
            
            expected_glossary=parse_dirty_glossary(item["glossary"])
            expected_padchheda = list(expected_glossary.keys())

            
            generated_reasoning_str=response_str.split("### FINAL GENERATION:")[0].strip().strip("\n")
            pattern = r'\[[^\]]*\]'
            generated_padacheda_list=re.findall(pattern, generated_reasoning_str)

            generated_glossary_str=response_str.split("### FINAL GENERATION:")[-1].strip().strip("\n")

            try:
                if generated_glossary_str.startswith("```json"):
                    generated_glossary_str = generated_glossary_str[7:]
                if generated_glossary_str.startswith("```python"):
                    generated_glossary_str = generated_glossary_str[10:]
                if generated_glossary_str.endswith("```"):
                    generated_glossary_str = generated_glossary_str[:-3]
                generated_dict = json.loads(clean_response_str)
                if "glossary" in generated_dict:
                    generated_dict = generated_dict["glossary"]
            except json.JSONDecodeError:
                try:
                    generated_dict = ast.literal_eval(generated_glossary_str)
                    if "glossary" in generated_dict:
                        generated_dict = generated_dict["glossary"]
                except (ValueError, SyntaxError):
                    generated_dict = generated_glossary_str
            
            step_result = {
                "shloka": target_shloka,
                "translation":item['translation'],
                "expected_padaccheda" : expected_padchheda,
                "expected_glossary": expected_glossary,
                "generated_padaccheda":generated_padacheda_list,
                "generated_glossary": generated_dict,
                "raw_generated_text": response_str 
            }
            with open(output_file_1b, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_result, ensure_ascii=False) + "\n")

    print("N-shot vLLM Inference complete!!")
