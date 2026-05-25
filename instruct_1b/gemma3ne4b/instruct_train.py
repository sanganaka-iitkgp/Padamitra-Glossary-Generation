import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig, Gemma3nForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from huggingface_hub import login
import os
import time


def wait_for_vram(target_gb=25, device=0, check_interval=60):
    target_bytes = target_gb * (1024 ** 3) 
    print(f"Waiting for {target_gb}GB of free VRAM to become available...")
    
    while True:
        free_vram, total_vram = torch.cuda.mem_get_info(device)
        
        if free_vram >= target_bytes:
            print(f"\nSuccess! {free_vram / (1024**3):.2f}GB free. Firing up the model!")
            break
            
        print(f"Current free VRAM: {free_vram / (1024**3):.2f}GB. Retrying in {check_interval}s...", end='\r')
        time.sleep(check_interval)


local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}
wait_for_vram(target_gb=36, device=local_rank, check_interval=15)

login()

model_id="unsloth/gemma-3n-E4B-it"
train_dataset = Dataset.load_from_disk("../data/pc_data_instruct_train_1b")
eval_dataset = Dataset.load_from_disk("../data/pc_data_instruct_eval_1b")

torch.manual_seed(42)

processor = AutoProcessor.from_pretrained(model_id)

processor.padding_side = "right"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["prediction_coefs", "altup", "correction_coefs","lm_head"]
)

print(f"Loading model on GPU {local_rank} in 4-bit...")
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config, 
    attn_implementation="eager"
)

peft_config = LoraConfig(
    r=256,            
    lora_alpha=16,        
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True     
)

training_args = SFTConfig(
    report_to="wandb",        
    run_name="unsloth-gemma-3ne4b-it-glossary-1b-v1",

    output_dir="./unsloth-gemma-3ne4b-it-glossary-1b",
    # output_dir="./temp-test",
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8,
    
    optim="adamw_torch_fused", 
    logging_steps=10,
    
    save_strategy="steps",     
    save_steps=600,      ##160
    eval_strategy="steps",     
    eval_steps=600,

    bf16=True, 
    max_grad_norm=1.0, 

    learning_rate=3e-5, 
    warmup_ratio=0.15,
    lr_scheduler_type="cosine",
    
    seed=42, 

    num_train_epochs=10,               
    load_best_model_at_end=True,       
    metric_for_best_model="eval_loss", 
    greater_is_better=False,

    save_total_limit=2,
    save_only_model=True,
    max_length=1024,
    
    completion_only_loss=True,
    
    # ddp_find_unused_parameters=True,
    gradient_checkpointing=False
)

trainer = SFTTrainer( 
    model=model, 
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,   
    args=training_args,
    peft_config=peft_config,
    processing_class=processor, 
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001 
    )] 
)

batch = trainer.data_collator([trainer.train_dataset[0]])
full_text = processor.decode(batch["input_ids"][0], skip_special_tokens=False)
print("=== FULL TRAINING TEXT ===")
print(full_text)

label_ids = batch["labels"][0]
completion_tokens = [t for t in label_ids if t != -100]
print("\n=== COMPLETION TOKENS (what model is trained to predict) ===")
print(processor.decode(completion_tokens, skip_special_tokens=False))


trainer.train()

print("Saving the best model...")
trainer.save_model("./unsloth-gemma-3ne4b-it-glossary-1b-best-model")
processor.save_pretrained("./unsloth-gemma-3ne4b-it-glossary-1b-best-model")
print("Training complete and model saved!")

