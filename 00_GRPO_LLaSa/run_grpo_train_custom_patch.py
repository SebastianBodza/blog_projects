import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' We can't limit the GPUs with the env var as vllm needs to move to 1

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
import os 
import asyncio
import aiohttp
import re 
from filter_text import filter_text

load_dotenv("../.env")

os.environ["WANDB_PROJECT"] = "SmolKartoffel_GRPO"

model_name = "SebastianBodza/SmolKartoffel-135M-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    token=os.getenv("HF_TOKEN"))

# Load LoRA
# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=16,
#     lora_alpha=32,
#     target_modules="all-linear",
#     lora_dropout=0.05,

# )
# model = get_peft_model(model, lora_config)
# print(model.print_trainable_parameters())

processor = AutoTokenizer.from_pretrained(model_name)


dataset = load_from_disk("/media/bodza/Audio_Dataset/COMPLETE_DATASET")
dataset = dataset["train"].shuffle(43).select(range(50_000)).filter(filter_text, num_proc=4, batched=True)
#     features: ['text', 'start', 'end', 'speaker', 'language', 
# 'dnsmos', 'podcast_name', 'episode_name', 'utterance_pitch_mean', 
# 'utterance_pitch_std', 'snr', 'c50', 'speaking_rate', 'phonemes', 
# 'gender', 'stoi', 'si-sdr', 'pesq', 'speaker_id', 'pitch', 'noise', 
# 'reverberation', 'speech_monotony', 'sdr_noise', 'pesq_speech_quality', 'prompt', 
# 'text_description', 'answer']
dataset = dataset.shuffle(seed=43).select(range(10_000)).map(
    lambda x: {
        "prompt": [
        {"role": "user", "content": "Convert the text to speech:" + "<|TEXT_UNDERSTANDING_START|>{}<|TEXT_UNDERSTANDING_END|>".format(x["text"])},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
    ], 
        "answer": x["text"],
    }
)
print(dataset)

async def process_audio_sample(speech_tokens_str: str, expected_answer: str) -> float:
    """
    Process a single sample by calling the combined /end_to_end endpoint,
    which decodes speech tokens and transcribes the resulting audio.
    The service returns (among others) the transcribed text, WER, and reward.
    """
    payload = {
        "speech_tokens_str": speech_tokens_str,
        "expected_text": expected_answer,
    }
    transcribed_text = ""
    wer = None
    reward = 0.0

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/end_to_end", json=payload, timeout=300
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    transcribed_text = result.get("transcribed_text", "")
                    wer = result.get("wer", None)
                    reward = result.get("reward", 0.0)
                else:
                    error_text = await response.text()
                    print(
                        f"Error in combined endpoint: {response.status} - {error_text}"
                    )
    except Exception as e:
        print(f"Exception in combined endpoint request: {e}")

    print(
        "-" * 20,
        f"\nExpected Answer:\n{expected_answer}",
        f"\nTranscribed:\n{transcribed_text}",
        f"\nWER: {wer}, Reward: {reward}",
        f"\nWER Reward: {result.get('wer_reward', None)}, Hangup Reward: {result.get('hangup_reward', None)}",
    )
    return reward



async def wer_reward_func_async(
    speech_tokens_list: list[str], answers: list[str]
) -> list[float]:
    """
    Async version of the reward function that processes all samples in
    parallel using the combined endpoint.
    """
    tasks = [
        process_audio_sample(speech_tokens, answer)
        for speech_tokens, answer in zip(speech_tokens_list, answers)
    ]
    rewards = await asyncio.gather(*tasks)
    return rewards


def wer_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Synchronous interface for the async reward function.
    Processes all transcription requests in parallel using the combined endpoint.
    Expects the completions to be a list where each element is a list/dict
    that contains the speech token string in completion[0]['content'].
    """
    speech_tokens_list = [completion[0]["content"] for completion in completions]
    return asyncio.run(wer_reward_func_async(speech_tokens_list, answer))




output_dir = "/media/bodza/Audio_Dataset/llasa_GRPO_higherAcc_cosine_higherlr_longer_cosine_longer"
run_name = "SmolKartoffel-135M-v0.1_GRPO_higherAcc_cosine_higherlr_longer_cosine_longer"

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.05,
    # lr_scheduler_type='cosine',
    lr_scheduler_type="constant_with_warmup",
    optim = "paged_adamw_8bit",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps=8,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=2048,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.2,
    report_to="wandb",
    log_on_each_node=False,
    use_vllm=True,
    # vllm_device="cuda:0",
    # vllm_gpu_memory_utilization=0.08,
    vllm_device="cuda:1",
    vllm_gpu_memory_utilization=0.3,
    vllm_max_model_len=2048,
    vllm_dtype="half",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # use_liger_kernel=True did not work for me
    # num_iterations=2
    )

processor.pad_token = processor.eos_token

trainer = GRPOTrainer(
    model=model, 
    processing_class=processor,
    reward_funcs=[wer_reward_func],
    args=training_args,
    train_dataset=dataset,    
    # peft_config=lora_config
    )


trainer.train()