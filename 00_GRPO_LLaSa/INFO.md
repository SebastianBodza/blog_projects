# SmolLM-GRPO: German Speech Generation with Reinforcement Learning

A weekend project experimenting with GRPO using a fine-tuned 135M parameter German SmolLM model for speech generation.

## Project Overview

This project implements reinforcement learning to improve speech generation quality in a small German TTS model. By using a custom reward function that combines Word Error Rate (WER) and repetition detection, the model learns to generate more stable speech.

## Technical Architecture

- **Base Model**: Fine-tuned 135M parameter German SmolLM model for Llasa style (Llama-style architecture)
- **Training Method**: GRPO
- **Hardware**: Dual RTX 3090 GPUs
- **Training Time**: ~13 hours

### Component Distribution

- **GPU 0**: GRPO training process
- **GPU 1**: vLLM inference and FastAPI server
- **FastAPI Services**: 
  - xcodec decoding
  - Whisper model for speech recognition

## Reward Function

The reward is computed as the mean of two components:
1. **Word Error Rate**: `math.exp(-wer)` - Rewards accuracy of speech
2. **Repetition Penalty**: Binary score (1 if no repetition detected, 0 otherwise)

## Implementation Notes

- FastAPI was chosen to decouple components without the complexity of multiprocessing with CUDA devices
- Architecture allows for flexible deployment across different hardware
- Added `.to()` statements in the GRPOTrainer to resolve CUDA device allocation errors. The updated GRPO with vLLM detached inference is advisable

## Results and Observations

The reinforcement learning process yielded both expected and unexpected outcomes:

- ✅ **Improved speech generation quality** by being more stable
- ⚠️ **Emergent behavior**: The model learned to speak faster when generating longer texts
- **Why?** The model discovered that speaking faster helps avoid audio truncation at the 2k token limit, resulting in higher rewards

This demonstrates how rl systems can develop creative solutions to maximize their reward functions. Just like the usual RL example of the boat driving in circles. 

## Future Improvements

- Adjust reward function to e.g. similarity of the audio, speaker similarity, ... 

