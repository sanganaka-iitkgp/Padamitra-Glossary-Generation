import json
import re
import ast
import os
import subprocess
from datasets import load_dataset
from collections import defaultdict

def filter_sans_to_english(dataset):
    return dataset.filter(lambda x: x['source_language'] == 'sanskrit' 
                                    and x['target_language'] == 'english')

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

def clean_shloka_for_alignment(shloka: str) -> str:
    if not shloka:
        return ""
    shloka = shloka.replace("।।", " ").replace("।", " ")
    shloka = re.sub(r'[^\w\s\u0900-\u097F]', ' ', shloka)
    
    return " ".join(shloka.split())

def clean_translation_for_alignment(translation: str) -> str:
    if not translation:
        return ""
    translation = translation.lower()
    translation = re.sub(r'[^\w\s]', ' ', translation)
    return " ".join(translation.split())

def prepare_alignment_files(train_dataset, test_dataset, output_dir="./alignment_data"):
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/train.sa", "w", encoding="utf-8") as f_sa, \
         open(f"{output_dir}/train.en", "w", encoding="utf-8") as f_en:
        for row in train_dataset:
            sa = clean_shloka_for_alignment(row['shlok'])
            en = clean_translation_for_alignment(row['translation'])
            f_sa.write(sa + "\n")
            f_en.write(en + "\n")

    test_start_idx = len(train_dataset)

    with open(f"{output_dir}/corpus.sa", "w", encoding="utf-8") as f_sa, \
         open(f"{output_dir}/corpus.en", "w", encoding="utf-8") as f_en:

        for row in train_dataset:
            sa = clean_shloka_for_alignment(row['shlok'])
            en = clean_translation_for_alignment(row['translation'])
            f_sa.write(sa + "\n")
            f_en.write(en + "\n")

        for row in test_dataset:
            sa = clean_shloka_for_alignment(row['shlok'])
            en = clean_translation_for_alignment(row['translation'])
            f_sa.write(sa + "\n")
            f_en.write(en + "\n")

    print(f"Train pairs : {len(train_dataset)}")
    print(f"Test pairs  : {len(test_dataset)}")
    print(f"Test starts at line: {test_start_idx}")
    print(f"Files saved to {output_dir}/")

    return test_start_idx



def run_fastalign(output_dir="./alignment_data"):
    fa_input = f"{output_dir}/fastalign_input.txt"
    with open(f"{output_dir}/corpus.sa", encoding="utf-8") as f_sa, \
         open(f"{output_dir}/corpus.en", encoding="utf-8") as f_en, \
         open(fa_input, "w", encoding="utf-8") as f_out:
        for sa, en in zip(f_sa, f_en):
            f_out.write(f"{sa.strip()} ||| {en.strip()}\n")

    fa_output_fwd = f"{output_dir}/fastalign_fwd.align"
    cmd_fwd = f"fast_align -i {fa_input} -d -o -v > {fa_output_fwd}"
    print("Running FastAlign (forward)...")
    os.system(cmd_fwd)

    
    fa_output_rev = f"{output_dir}/fastalign_rev.align"
    cmd_rev = f"fast_align -i {fa_input} -d -o -v -r > {fa_output_rev}"
    print("Running FastAlign (reverse)...")
    os.system(cmd_rev)

    
    fa_output_sym = f"{output_dir}/fastalign_sym.align"
    cmd_sym = f"atools -i {fa_output_fwd} -j {fa_output_rev} -c grow-diag-final-and > {fa_output_sym}"
    print("Running symmetrization...")
    os.system(cmd_sym)

    print(f"FastAlign alignments saved to {fa_output_sym}")
    return fa_output_sym




def alignment_to_glossary(alignment_line: str, sa_words: list, en_words: list) -> dict:
    glossary = {}
    pairs = alignment_line.strip().split()

    
    sa_to_en = defaultdict(list)
    for pair in pairs:
        if '-' not in pair:
            continue
        sa_idx, en_idx = map(int, pair.split('-'))
        if sa_idx < len(sa_words) and en_idx < len(en_words):
            sa_to_en[sa_idx].append(en_words[en_idx])

    for sa_idx, en_word_list in sorted(sa_to_en.items()):
        sa_word = sa_words[sa_idx]
        en_meaning = " ".join(en_word_list)
        glossary[sa_word] = en_meaning

    return glossary


def extract_glossaries_from_alignments(align_file: str,
                                        corpus_sa: str,
                                        corpus_en: str,
                                        test_start_idx: int,
                                        test_dataset,
                                        output_jsonl: str):
    """Extract glossaries for test set only and save to jsonl."""

    with open(align_file, encoding="utf-8") as f_align, \
         open(corpus_sa,  encoding="utf-8") as f_sa, \
         open(corpus_en,  encoding="utf-8") as f_en:

        alignments   = f_align.readlines()
        sa_sentences = f_sa.readlines()
        en_sentences = f_en.readlines()

    results = []
    for i, row in enumerate(test_dataset):
        corpus_idx = test_start_idx + i

        sa_words   = sa_sentences[corpus_idx].strip().split()
        en_words   = en_sentences[corpus_idx].strip().split()
        align_line = alignments[corpus_idx]

        generated_glossary = alignment_to_glossary(align_line, sa_words, en_words)
        expected_glossary  = parse_dirty_glossary(row['glossary'])

        results.append({
            "shloka":              row['shlok'],
            "translation":         row['translation'],
            "expected_glossary":   expected_glossary,
            "generated_glossary":  generated_glossary,
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} glossaries to {output_jsonl}")




if __name__ == "__main__":
    from huggingface_hub import login
    login()

    dataset      = load_dataset("sanganaka/padamitra")
    train_dataset = filter_sans_to_english(dataset["train"])
    test_dataset  = filter_sans_to_english(dataset["test"])

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    
    test_start_idx = prepare_alignment_files(train_dataset, test_dataset)

    
    align_file = run_fastalign()

    
    extract_glossaries_from_alignments(
        align_file       = align_file,
        corpus_sa        = "./alignment_data/corpus.sa",
        corpus_en        = "./alignment_data/corpus.en",
        test_start_idx   = test_start_idx,
        test_dataset     = test_dataset,
        output_jsonl     = "./fastalign_glossary_results.jsonl"
    )

    print("Done! Run your eval script on fastalign_glossary_results.jsonl")