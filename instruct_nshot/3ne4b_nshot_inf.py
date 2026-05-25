import json
import re
import ast
import time
import torch
from datasets import load_dataset
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


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


# wait_for_vram(target_gb=30, device=0, check_interval=10)


login()

model_id = "unsloth/gemma-3n-E4B-it"
model_name = "gemma-3ne4b"

N_SHOTS = [1,5,10]


# ─────────────────────────────────────────────
# Dataset helpers  (unchanged)
# ─────────────────────────────────────────────

def filter_sans_to_english(dataset):
    return dataset.filter(
        lambda example: example['source_language'] == 'sanskrit'
        and example['target_language'] == 'english'
    )


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
                next_key = chunk[last_comma_idx + 1:].strip()

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

    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
    if s.endswith("```"):
        s = s[:-3]

    s = s.strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        return s


# ─────────────────────────────────────────────
# Prompt builder  (unchanged – exact same prompts)
# ─────────────────────────────────────────────

def build_n_shot_messages(target_shloka, target_trans, train_dataset, n_shots=3):
    messages_1a, messages_1b = [], []

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
        print(f"Shlok: {shlok}")
        print(f"Translation: {translation}")
        print(f"Glossary: {clean_glossary}\n\n")

        resolved_sequence = list(clean_glossary.keys())
        if not resolved_sequence:
            print("Oh my habibiii")

        prompt_1a = (
            f"Generate a Sanskrit-English glossary mapping for the following shloka based on its translation.\n\n"
            f"Shloka:\n{shlok}\n\nTranslation:\n{translation}"
        )
        assistant_response_1a = f'{{"glossary": {clean_glossary}}}'

        sequence_string = ", ".join(resolved_sequence)
        prompt_1b = (
            f"### INPUT:\nYou are an expert in Sanskrit. Perform Padaccheda (word resolution) on the following Sanskrit shloka, "
            f"then generate the glossary mapping using the generated Padaccheda and provided English translation.\n\n"
            f"Shloka:\n{shlok}\n\nTranslation:\n{translation}"
        )
        reasoning_text = (
            "To build the glossary, I first need to divide the shloka into its semantic parts. "
            "By analyzing the given shloka and referencing the translation, I will break down the sandhi "
            "while keeping samasa (compound words) intact. "
            f"Upon dividing this, the resolved sequence of words is: [{sequence_string}]."
        )
        glossary_json = json.dumps({"glossary": clean_glossary}, ensure_ascii=False)
        assistant_response_1b = (
            f"### REASONING:\n{reasoning_text}\n\n"
            f"### FINAL GENERATION:\n```json\n{glossary_json}\n```"
        )

        messages_1a.append({"role": "user",      "content": prompt_1a})
        messages_1a.append({"role": "assistant",  "content": assistant_response_1a})

        messages_1b.append({"role": "user",      "content": prompt_1b})
        messages_1b.append({"role": "assistant",  "content": assistant_response_1b})

    target_prompt_1a = (
        f"Generate a Sanskrit-English glossary mapping for the following shloka based on its translation.\n\n"
        f"Shloka:\n{target_shloka}\n\nTranslation:\n{target_trans}"
    )
    target_prompt_1b = (
        f"### INPUT:\nYou are an expert in Sanskrit. Perform Padaccheda (word resolution) on the following Sanskrit shloka, "
        f"then generate the glossary mapping using the generated Padaccheda and provided English translation.\n\n"
        f"Shloka:\n{target_shloka}\n\nTranslation:\n{target_trans}"
    )

    messages_1a.append({"role": "user", "content": target_prompt_1a})
    messages_1b.append({"role": "user", "content": target_prompt_1b})

    return messages_1a, messages_1b


# ─────────────────────────────────────────────
# Single-sample inference helper
# ─────────────────────────────────────────────

def run_inference(model, tokenizer, messages, max_new_tokens=1200, device="cuda"):
    """Tokenise a chat conversation, run greedy decoding, return generated text only."""
    # Step 1: format the chat into a single string (no tokenization yet)
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    # Step 2: tokenize the string normally — always returns a proper tensor
    tokenized = tokenizer(prompt_str, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy (mirrors temperature=0 in vLLM)
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens so we return only the generated portion
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer and model for {model_id}...")
    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",          # spreads across available GPUs automatically
    #     trust_remote_code=True,
    # )
    # model.eval()
    # print("Model loaded.")

    dataset = load_dataset("sanganaka/padamitra")
    train_dataset = filter_sans_to_english(dataset["train"])
    test_dataset  = filter_sans_to_english(dataset["test"])
    test_dataset  = test_dataset.select(range(1))

    for shot in N_SHOTS:
        print(f"\n{'='*60}")
        print(f"Starting {shot}-shot inference...")
        print(f"{'='*60}")

        # output_file_1a = f"1a/{model_name}/{model_name}_hf_{shot}_shot_results.jsonl"
        # output_file_1b = f"1b/{model_name}/{model_name}_hf_{shot}_shot_results.jsonl"

        # # Truncate / create output files
        # open(output_file_1a, "w").close()
        # open(output_file_1b, "w").close()

        for idx, item in enumerate(tqdm(test_dataset, desc=f"{shot}-shot")):
            target_shloka = item["shlok"]
            target_trans  = item["translation"]

            messages_1a, messages_1b = build_n_shot_messages(
                target_shloka, target_trans, train_dataset, n_shots=shot
            )

            # # ── 1a inference ──────────────────────────────────────────
            # response_str_1a = run_inference(model, tokenizer, messages_1a, device=device)

            # clean_1a = response_str_1a.strip()
            # if clean_1a.startswith("```json"):
            #     clean_1a = clean_1a[7:]
            # if clean_1a.startswith("```python"):
            #     clean_1a = clean_1a[9:]
            # if clean_1a.endswith("```"):
            #     clean_1a = clean_1a[:-3]
            # clean_1a = clean_1a.strip()

            # try:
            #     generated_dict_1a = json.loads(clean_1a)
            #     if "glossary" in generated_dict_1a:
            #         generated_dict_1a = generated_dict_1a["glossary"]
            # except json.JSONDecodeError:
            #     try:
            #         generated_dict_1a = ast.literal_eval(clean_1a)
            #         if "glossary" in generated_dict_1a:
            #             generated_dict_1a = generated_dict_1a["glossary"]
            #     except (ValueError, SyntaxError):
            #         generated_dict_1a = clean_1a

            # result_1a = {
            #     "shloka":            target_shloka,
            #     "translation":       target_trans,
            #     "expected_glossary": parse_dirty_glossary(item["glossary"]),
            #     "generated_glossary": generated_dict_1a,
            #     "raw_text":          response_str_1a,
            # }
            # with open(output_file_1a, "a", encoding="utf-8", errors="replace") as f:
            #     f.write(json.dumps(result_1a, ensure_ascii=False) + "\n")

            # # ── 1b inference ──────────────────────────────────────────
            # response_str_1b = run_inference(model, tokenizer, messages_1b, device=device)

            # expected_glossary  = parse_dirty_glossary(item["glossary"])
            # expected_padchheda = list(expected_glossary.keys())

            # generated_reasoning_str = response_str_1b.split("### FINAL GENERATION:")[0].strip().strip("\n")
            # pattern = r'\[[^\]]*\]'
            # generated_padacheda_list = re.findall(pattern, generated_reasoning_str)

            # generated_glossary_str = response_str_1b.split("### FINAL GENERATION:")[-1].strip().strip("\n")

            # try:
            #     if generated_glossary_str.startswith("```json"):
            #         generated_glossary_str = generated_glossary_str[7:]
            #     if generated_glossary_str.startswith("```python"):
            #         generated_glossary_str = generated_glossary_str[9:]
            #     if generated_glossary_str.endswith("```"):
            #         generated_glossary_str = generated_glossary_str[:-3]

            #     # Try parsing the cleaned glossary substring first
            #     generated_dict_1b = json.loads(generated_glossary_str.strip())
            #     if "glossary" in generated_dict_1b:
            #         generated_dict_1b = generated_dict_1b["glossary"]
            # except json.JSONDecodeError:
            #     try:
            #         generated_dict_1b = ast.literal_eval(generated_glossary_str.strip())
            #         if "glossary" in generated_dict_1b:
            #             generated_dict_1b = generated_dict_1b["glossary"]
            #     except (ValueError, SyntaxError):
            #         generated_dict_1b = generated_glossary_str

            # result_1b = {
            #     "shloka":               target_shloka,
            #     "translation":          target_trans,
            #     "expected_padaccheda":  expected_padchheda,
            #     "expected_glossary":    expected_glossary,
            #     "generated_padaccheda": generated_padacheda_list,
            #     "generated_glossary":   generated_dict_1b,
            #     "raw_generated_text":   response_str_1b,
            # }
            # with open(output_file_1b, "a", encoding="utf-8", errors="replace") as f:
            #     f.write(json.dumps(result_1b, ensure_ascii=False) + "\n")

    print("\nN-shot HuggingFace inference complete!!")
