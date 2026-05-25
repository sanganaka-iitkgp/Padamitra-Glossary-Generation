import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig, Gemma3nForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from huggingface_hub import login
import os

login()

base_model_id = "google/gemma-3-12b-it"
train_dataset = Dataset.load_from_disk("../data/data_instruct_train_1a")
eval_dataset = Dataset.load_from_disk("../data/data_instruct_eval_1a")


torch.manual_seed(42)

processor = AutoProcessor.from_pretrained(model_id)


processor.padding_side = "right"

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True
)

print(f"Loading model on GPU {local_rank} in 4-bit...")
model = Gemma3nForConditionalGeneration.from_pretrained(
    model_id,
    device_map=device_map,
    # device_map="auto",
    quantization_config=bnb_config, 
    attn_implementation="sdpa"
)

peft_config = LoraConfig(
    r=256,            
    lora_alpha=16,        
    target_modules="all-linear",  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True     
)
# model=get_peft_model(model,peft_config)
# print(model.get_nb_trainable_parameters())

training_args = SFTConfig(
    report_to="wandb",        
    run_name="gemma-3-12b-it-glossary-1a-checkpoints",

    output_dir="./gemma-e4b-it-glossary-1a",
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=2,
    
    optim="adamw_torch_fused", 
    logging_steps=10,
    
    save_strategy="steps",     
    save_steps=300,      ##160
    eval_strategy="steps",     
    eval_steps=300,

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

    save_total_limit=4,
    max_length=512,
          
    # assistant_only_loss=True,
    completion_only_loss=True,
    ddp_find_unused_parameters=True,
    gradient_checkpointing=True
)

# class Gemma3SFTTrainer(SFTTrainer):
#     def _prepare_inputs(self, inputs):
#         inputs = super()._prepare_inputs(inputs)
        
#         if "token_type_ids" not in inputs and "input_ids" in inputs:
#             inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
            
#         return inputs

trainer = SFTTrainer( 
    model=model,                  
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,   
    args=training_args,
    peft_config=peft_config,
    processing_class=processor, 
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=8,
        early_stopping_threshold=0.001 
    )] 
)

trainer.train()

print("Saving the best model...")
trainer.save_model("./gemma-3-12b-it-glossary-1a-best-model")
processor.save_pretrained("./gemma-3-12b-it-glossary-1a-best-model")
print("Training complete and model saved!")
