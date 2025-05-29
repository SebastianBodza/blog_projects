import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import torch
from torch import nn
from torch.utils.data import Dataset # DataLoader removed as Trainer handles it
from transformers import Trainer, TrainingArguments
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from datasets import load_from_disk

# AttrDict can be useful for mimicking model output structures if needed
# but T3.loss returns a tuple, so it's less critical here.
# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self

class TTSDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "text_tokens": torch.LongTensor(sample["text_tokens"]),
            "speech_tokens": torch.LongTensor(sample["speech_tokens"]),
            "speaker_embed": torch.FloatTensor(sample["speaker_embed"]).squeeze(),
            "cond_prompt_speech_tokens": torch.LongTensor(sample["cond_prompt_speech_tokens"]).squeeze(),
            "emotion_adv": torch.FloatTensor([sample.get("emotion_adv", 0.5)]), # Ensure (1,)
        }

class T3DataCollator:
    def __init__(self):
        self.hp = T3Config()
        self.speech_cond_prompt_len = self.hp.speech_cond_prompt_len
        
        # Use a dedicated padding token ID. T3.loss uses IGNORE_ID = -100 internally.
        # It's good practice for the embedding layer to know about a pad_token_id if it's not 0.
        # However, T3.loss masks based on length, so the actual padding value fed to T3.loss
        # for text_tokens/speech_tokens doesn't strictly matter as long as lengths are correct.
        # For simplicity and consistency with many models, let's use 0 for padding if not specified.
        # T3Config stop_text_token is 0, stop_speech_token is 6562.
        # Let's define distinct pad tokens if T3.loss doesn't handle 0 as pad for inputs.
        # T3.loss uses IGNORE_ID = -100 for loss calculation.
        # The input padding value to T3.loss for text/speech tokens can be anything,
        # as long as text_token_lens/speech_token_lens are correct.
        # Let's use 0 for padding for simplicity, assuming embedding layers handle it or it's masked out.
        self.text_pad_token = 0 # Or a dedicated pad_id if your tokenizer/model expects it
        self.speech_pad_token = 0 # Or a dedicated pad_id

    def __call__(self, batch):
        # This collator will produce batches of size `per_device_train_batch_size`
        # The CFG duplication (batch of 2) will happen inside T3Trainer.compute_loss

        processed_batch_items = []
        for item in batch:
            # Add BOT/EOT tokens
            text = torch.cat([
                torch.LongTensor([self.hp.start_text_token]),
                item["text_tokens"],
                torch.LongTensor([self.hp.stop_text_token])
            ])
            speech = torch.cat([
                torch.LongTensor([self.hp.start_speech_token]),
                item["speech_tokens"],
                torch.LongTensor([self.hp.stop_speech_token])
            ])
            # Truncate cond_prompt_speech_tokens
            cond_prompt = item["cond_prompt_speech_tokens"][:self.speech_cond_prompt_len]
            
            processed_batch_items.append({
                "text_tokens": text,
                "speech_tokens": speech,
                "cond_prompt_speech_tokens": cond_prompt,
                "speaker_embed": item["speaker_embed"], # Shape (D_spk)
                "emotion_adv": item["emotion_adv"]      # Shape (1)
            })

        # Pad text_tokens
        text_tokens_list = [item["text_tokens"] for item in processed_batch_items]
        text_token_lens = torch.LongTensor([len(t) for t in text_tokens_list])
        text_padded = torch.nn.utils.rnn.pad_sequence(
            text_tokens_list, batch_first=True, padding_value=self.text_pad_token
        )

        # Pad speech_tokens
        speech_tokens_list = [item["speech_tokens"] for item in processed_batch_items]
        speech_token_lens = torch.LongTensor([len(s) for s in speech_tokens_list])
        speech_padded = torch.nn.utils.rnn.pad_sequence(
            speech_tokens_list, batch_first=True, padding_value=self.speech_pad_token
        )

        # Pad cond_prompt_speech_tokens
        cond_prompts_list = [item["cond_prompt_speech_tokens"] for item in processed_batch_items]
        # Ensure cond_prompts are padded to self.speech_cond_prompt_len if shorter, or truncated if longer
        # The model's prepare_conditioning expects a fixed length or handles variable if not using perceiver.
        # Here, we assume it's already truncated. We just need to pad to max_len in batch.
        cond_padded = torch.nn.utils.rnn.pad_sequence(
            cond_prompts_list, batch_first=True, padding_value=self.speech_pad_token # Use speech_pad_token for consistency
        )
        
        # Stack speaker embeddings and emotions
        speaker_embeds = torch.stack([item["speaker_embed"] for item in processed_batch_items]) # (B, D_spk)
        emotions = torch.stack([item["emotion_adv"] for item in processed_batch_items])         # (B, 1)

        return {
            "text_tokens": text_padded,
            "text_token_lens": text_token_lens,
            "speech_tokens": speech_padded,
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": speaker_embeds,
            "t3_cond_prompt_speech_tokens": cond_padded,
            "t3_cond_emotion_adv": emotions,
        }

class T3Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Inputs from collator (batch_size = per_device_train_batch_size)
        # Let B = per_device_train_batch_size
        text_tokens_cond = inputs["text_tokens"]                 # (B, L_txt_pad)
        text_token_lens_cond = inputs["text_token_lens"]         # (B,)
        speech_tokens_cond = inputs["speech_tokens"]             # (B, L_sp_pad)
        speech_token_lens_cond = inputs["speech_token_lens"]     # (B,)
        
        # Conditioning inputs (already batched by collator)
        t3_cond_obj = T3Cond(
            speaker_emb=inputs["t3_cond_speaker_emb"],                           # (B, D_spk)
            cond_prompt_speech_tokens=inputs["t3_cond_prompt_speech_tokens"],    # (B, L_prompt_pad)
            emotion_adv=inputs["t3_cond_emotion_adv"],                           # (B, 1)
            clap_emb=None, # Not used
            cond_prompt_speech_emb=None # Will be computed in model.prepare_conditioning
        )

        # --- Create the batch of 2 for CFG ---
        # The T3.forward method expects text_tokens[1] to be the unconditional part.
        
        # Unconditional text: just BOT, EOT, then padded.
        # Its embedding will be zeroed out by model.forward().
        hp = model.module.hp if hasattr(model, 'module') else model.hp # Handle DDP
        
        # Create unconditional text tokens (minimal: BOT, EOT, then pad)
        # The actual content of text_tokens_uncond doesn't matter much as its embedding is zeroed,
        # but its length (text_token_lens_uncond) is used for masking in T3.loss.
        text_tokens_uncond = torch.full_like(text_tokens_cond, 0) # Pad with 0 or a specific pad_id
        text_tokens_uncond[:, 0] = hp.start_text_token
        text_tokens_uncond[:, 1] = hp.stop_text_token
        text_token_lens_uncond = torch.full_like(text_token_lens_cond, 2) # Length is 2 (BOT, EOT)

        # Speech tokens for the unconditional part are the same as conditional
        # (we still want to predict the same speech, just without text conditioning)
        speech_tokens_uncond = speech_tokens_cond.clone()
        speech_token_lens_uncond = speech_token_lens_cond.clone()

        # Concatenate to form the CFG batch (size 2*B)
        # Note: T3.forward will expand t3_cond_obj if its batch size (B)
        # doesn't match text_tokens_cfg's batch size (2*B).
        text_tokens_cfg = torch.cat([text_tokens_cond, text_tokens_uncond], dim=0)
        text_token_lens_cfg = torch.cat([text_token_lens_cond, text_token_lens_uncond], dim=0)
        speech_tokens_cfg = torch.cat([speech_tokens_cond, speech_tokens_uncond], dim=0)
        speech_token_lens_cfg = torch.cat([speech_token_lens_cond, speech_token_lens_uncond], dim=0)
        
        # Call the model's own loss function
        # model.loss() will call model.forward() internally
        loss_text, loss_speech = model( # This directly calls T3.loss if model is T3 instance
            t3_cond=t3_cond_obj, # Will be expanded to 2*B inside model.forward
            text_tokens=text_tokens_cfg,
            text_token_lens=text_token_lens_cfg,
            speech_tokens=speech_tokens_cfg,
            speech_token_lens=speech_token_lens_cfg,
            # training=True is implicitly handled by model.loss calling model.forward
        )
        
        # T3.loss returns loss_text and loss_speech, calculated on the full CFG batch.
        # The text_loss for the unconditional part (where text_emb was zeroed) might be high
        # but contributes to training the text decoder.
        # The speech_loss for the unconditional part trains the model to generate speech
        # from conditioning only.
        
        # Combine losses (you can adjust weighting)
        # Original T3.loss calculates loss on the full CFG batch.
        # We can keep it simple and sum them, or weight them.
        # Let's assume a simple sum or average as the model's loss function might already be tuned for this.
        total_loss = (loss_text + loss_speech) / 2.0 # Or your preferred weighting
        
        # `return_outputs=True` is tricky here because model.loss doesn't return full outputs.
        # If you need full outputs, you'd call model.forward first, then compute loss.
        # For simplicity, we'll stick to just returning the loss.
        if return_outputs:
            # This part is harder to make simple if model.loss() is the primary call.
            # For now, let's just return the loss.
            # To get outputs, you'd need to call model.forward() separately and then model.loss()
            # or replicate the output structure.
            # For HF Trainer, only loss is strictly required from compute_loss.
            # If you need detailed logits for evaluation during training, this needs more work.
            # For now, we focus on getting the loss computation correct and simple.
            # Returning None for outputs when return_outputs=True is not standard.
            # Let's just return loss. Trainer handles the tuple (loss, outputs) if outputs are provided.
            # Since model.loss() doesn't return the full AttrDict, we can't easily provide it.
            # Simplest is to just return the loss.
            # If Trainer expects (loss, outputs_dict), we might need to return (total_loss, {})
            # For now, let's assume Trainer is fine with just the loss tensor.
             return (total_loss, {"loss_text": loss_text.detach(), "loss_speech": loss_speech.detach()})


        return total_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T3().to(device)

    dataset = TTSDataset("/media/bodza/Audio_Dataset/chatterbox_tts_dataset")
    collator = T3DataCollator()
    
    training_args = TrainingArguments(
        output_dir="./results_simplified", # New output dir
        num_train_epochs=50,
        per_device_train_batch_size=1, # Each call to compute_loss gets this many samples
                                       # CFG batch of 2 is created inside compute_loss
        gradient_accumulation_steps=8, # Effective batch size = 1 * 8 = 8 (then doubled to 16 for CFG)
        learning_rate=5e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        fp16=True, # Mixed precision
        save_strategy="steps", # Use "steps" for save_steps
        save_steps=1000, # Save more frequently initially
        logging_steps=10,
        remove_unused_columns=False, # Important for passing all dict keys
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        dataloader_num_workers=4, # If your system supports it
        report_to="wandb", # if you have wandb setup
        save_total_limit=3,
    )
    
    trainer = T3Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dataset size: {len(dataset)}")
    
    trainer.train()
    
    model.save_pretrained("./results_simplified/final_model_t3") # Use save_pretrained for HF models
    
if __name__ == "__main__":
    train()
