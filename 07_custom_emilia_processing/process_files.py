#!/usr/bin/env python3
import os
import glob
import random
import json
import torch
import ffmpeg
import tempfile
import time
import gc
import math
import argparse
import fcntl
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
import whisperx
from whisperx.audio import N_SAMPLES, log_mel_spectrogram


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process audio files with WhisperX")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (0, 1, etc.)")
    parser.add_argument(
        "--sample-size", type=int, default=10, help="Number of files to process"
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=0,
        choices=[0, 1],
        help="Instance ID (0 or 1) to split workload",
    )
    return parser.parse_args()


# Parse arguments
args = parse_args()

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print(f"Using GPU: {args.gpu}")

# Load environment variables
load_dotenv()
os.environ["HF_HOME"] = "/media/bodza/Audio_Dataset/hf_cache/"
# Constants
PROCESSING_DIR = "/media/bodza/Audio_Dataset/RAW_PROCESSED"
OUTPUT_DIR = "/media/bodza/Audio_Dataset/MULTI_SPEAKER_PROCESSED/WHISPERX_NEW"
SAMPLE_SIZE = args.sample_size  # Number of files to process in one run
INSTANCE_ID = args.instance_id  # Instance ID (0 or 1)
LOCK_DIR = os.path.join(OUTPUT_DIR, "locks")

# WhisperX parameters
WHISPER_ARCH = "Systran/faster-whisper-large-v3"
COMPUTE_TYPE = "float16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
INITIAL_PROMPT = "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. hahaha. Oh. äh, hm, so, tja, halt, ähm, eigentlich. haha. euh."
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BEAM_SIZE = 5
BEST_OF = 5

VAD_ONSET = 0.500
VAD_OFFSET = 0.363
LANGUAGE_DETECTION_MIN_PROB = 0
LANGUAGE_DETECTION_MAX_TRIES = 5
ALIGN_OUTPUT = True
DIARIZATION = True
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MIN_SPEAKERS = None
MAX_SPEAKERS = None
DEBUG = True

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create lock directory if it doesn't exist
if not os.path.exists(LOCK_DIR):
    os.makedirs(LOCK_DIR)


def get_lock_path(file_path):
    """Get lock file path for a given audio file."""
    file_hash = str(hash(file_path) % 10000).zfill(5)
    return os.path.join(LOCK_DIR, f"lock_{file_hash}.lock")


def acquire_lock(lock_path):
    """Try to acquire lock for a file. Return True if successful, False otherwise."""
    try:
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write process ID to lock file
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        return True, lock_file
    except (IOError, BlockingIOError):
        return False, None


def release_lock(lock_file, lock_path):
    """Release the lock."""
    if lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    if os.path.exists(lock_path):
        os.unlink(lock_path)


def search_unprocessed_files():
    """Find all audio files that haven't been processed yet."""
    all_files = []
    for ext in ["*.mp3", "*.wav", "*.m4a", "*.mp4", "*.flac"]:
        pattern = os.path.join(PROCESSING_DIR, "**", ext)
        all_files.extend(glob.glob(pattern, recursive=True))

    unprocessed_files = []
    for file_path in all_files:
        base_name_with_path = os.path.splitext(file_path)[0]
        json_path = base_name_with_path + ".json"
        json_exists_and_not_empty = (
            os.path.exists(json_path) and os.path.getsize(json_path) > 0
        )

        # Check if output exists
        relative_path = os.path.relpath(os.path.dirname(file_path), PROCESSING_DIR)
        base_filename = os.path.basename(base_name_with_path)
        output_subdir = os.path.join(OUTPUT_DIR, relative_path, base_filename)
        output_mp3s_exist = os.path.exists(output_subdir) and glob.glob(
            os.path.join(output_subdir, "*.mp3")
        )

        # Check if file is locked (being processed by another instance)
        lock_path = get_lock_path(file_path)
        file_locked = os.path.exists(lock_path)

        if not json_exists_and_not_empty and not output_mp3s_exist and not file_locked:
            unprocessed_files.append(file_path)

    return unprocessed_files


def get_weighted_sample(unprocessed_files, sample_size):
    """
    Get a sample of files with equal representation from each folder,
    split based on instance ID.
    """
    if not unprocessed_files:
        return []

    # Group files by folder
    folders = {}
    for f in unprocessed_files:
        folder_name = os.path.basename(os.path.dirname(f))
        # Handle librivox special case
        if "librivox" in folder_name.lower():
            folder_name = "librivox"

        if folder_name not in folders:
            folders[folder_name] = []
        folders[folder_name].append(f)

    print(f"Found {len(folders)} unique folders")

    # Determine how many files to sample per folder
    num_folders = len(folders)
    files_per_folder = max(1, sample_size // num_folders)
    remaining = sample_size - (files_per_folder * num_folders)

    # Select files from each folder
    selected_files = []
    folder_names = sorted(list(folders.keys()))  # Sort to ensure consistent order

    # Process alternating folders based on instance ID
    # Instance 0 gets folders at even indices, Instance 1 gets folders at odd indices
    instance_folders = folder_names[INSTANCE_ID::2]

    # First ensure each folder gets represented
    for folder_name in instance_folders:
        folder_files = folders[folder_name]
        # Take min in case a folder has fewer files than files_per_folder
        num_to_select = min(
            files_per_folder * 2, len(folder_files)
        )  # Double the files since we're using half the folders
        if num_to_select > 0:
            selected_files.extend(random.sample(folder_files, num_to_select))

    # If we need more files to reach sample_size, randomly select from remaining files
    # (from the instance's assigned folders only)
    if remaining > 0 and len(selected_files) < sample_size:
        remaining_files = []
        for folder_name in instance_folders:
            remaining_files.extend(
                [f for f in folders[folder_name] if f not in selected_files]
            )

        if remaining_files:
            # Select additional files randomly
            additional = random.sample(
                remaining_files, min(remaining, len(remaining_files))
            )
            selected_files.extend(additional)

    # If we have too many files, randomly select the exact sample size
    if len(selected_files) > sample_size:
        selected_files = random.sample(selected_files, sample_size)

    # Print summary information
    print(
        f"\nSelected {len(selected_files)} files with uniform folder distribution (Instance {INSTANCE_ID}):"
    )
    folder_distribution = Counter(
        [os.path.basename(os.path.dirname(f)) for f in selected_files]
    )
    for folder, count in folder_distribution.most_common():
        print(f"  Folder: {folder} - Files: {count}")
    for f in selected_files:
        folder = os.path.basename(os.path.dirname(f))
        print(f"  - {folder}: {os.path.basename(f)}")
    return selected_files


def get_audio_duration(file_path):
    """Get duration of audio file in milliseconds."""
    probe = ffmpeg.probe(file_path)
    stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    if stream:
        return float(stream["duration"]) * 1000
    return 0


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    """Extract audio segment from file."""
    input_file_path = (
        Path(input_file_path)
        if not isinstance(input_file_path, Path)
        else input_file_path
    )
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        if DEBUG:
            print(f"Extracting from {input_file_path.name} to {temp_file.name}")

        try:
            (
                ffmpeg.input(input_file_path, ss=start_time_ms / 1000)
                .output(temp_file.name, t=duration_ms / 1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print("ffmpeg error occurred: ", e.stderr.decode("utf-8"))
            raise e

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    """Distribute time segments evenly across a duration."""
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def detect_language(
    full_audio_file_path,
    segments_starts,
    language_detection_min_prob,
    language_detection_max_tries,
    asr_options,
    vad_options,
    iteration=1,
):
    """Detect language from audio file, trying multiple segments if needed."""
    model = whisperx.load_model(
        WHISPER_ARCH,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    start_ms = segments_starts[iteration - 1]
    audio_segment_file_path = extract_audio_segment(
        full_audio_file_path, start_ms, 30000
    )
    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )

    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(
        f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})"
    )

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration,
    }

    if (
        language_probability >= language_detection_min_prob
        or iteration >= language_detection_max_tries
    ):
        return detected_language

    next_iteration_detected_language = detect_language(
        full_audio_file_path,
        segments_starts,
        language_detection_min_prob,
        language_detection_max_tries,
        asr_options,
        vad_options,
        iteration + 1,
    )

    if (
        next_iteration_detected_language["probability"]
        > detected_language["probability"]
    ):
        return next_iteration_detected_language

    return detected_language


def align(audio, result, debug):
    """Align transcription with audio."""
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    """Perform speaker diarization."""
    start_time = time.time_ns() / 1e6

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=huggingface_access_token, device=DEVICE
    )
    diarize_segments = diarize_model(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result


def concat_speaker_samples(asr_result, max_duration=60, max_speaker_gap=4):
    """
    Concatenate ASR segments into samples with alternating speakers.
    """
    samples = []
    current_sample = []
    last_end = None
    last_speaker = None
    sample_start = None
    sample_end = None

    for seg in asr_result:
        # Ensure segments have necessary keys
        if not all(k in seg for k in ["start", "end", "text"]):
            print(f"Skipping segment due to missing keys: {seg}")
            continue

        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        if not text:  # Skip segments with no text
            continue

        # If this is the first segment in the sample
        if not current_sample:
            current_sample.append(f"[{speaker}]: {text}")
            sample_start = seg["start"]
            sample_end = seg["end"]
            last_end = seg["end"]
            last_speaker = speaker
            continue  # Move to the next segment

        # Check conditions to finish the current sample
        gap = seg["start"] - last_end
        gap_exceeded = gap > max_speaker_gap
        potential_timespan = seg["end"] - sample_start
        duration_exceeded = potential_timespan > max_duration

        if gap_exceeded or duration_exceeded:
            # Finish current sample
            if current_sample:
                samples.append(
                    {
                        "text": " ".join(current_sample),
                        "start": sample_start,
                        "end": sample_end,
                        "filename": None,
                    }
                )

            # Start new sample with the current segment
            current_sample = [f"[{speaker}]: {text}"]
            sample_start = seg["start"]
            sample_end = seg["end"]
            last_end = seg["end"]
            last_speaker = speaker
        else:
            # Continue current sample
            if speaker != last_speaker:
                current_sample.append(f"[{speaker}]: {text}")
            else:
                # Same speaker, just append text to the last entry
                if current_sample:
                    current_sample[-1] += " " + text
                else:
                    current_sample.append(f"[{speaker}]: {text}")

            sample_end = seg["end"]
            last_end = seg["end"]
            last_speaker = speaker

    # Add the very last sample if it exists
    if current_sample:
        samples.append(
            {
                "text": " ".join(current_sample),
                "start": sample_start,
                "end": sample_end,
                "filename": None,
            }
        )

    for i, sample in enumerate(samples):
        sample["filename"] = f"segment_{i:04d}.mp3"

    return samples


def extract_and_save_segments(audio_file_path, segments, output_dir):
    """Extract and save audio segments."""
    os.makedirs(output_dir, exist_ok=True)

    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        duration = end_time - start_time

        # Create output file path
        output_filename = segment["filename"]
        output_file = os.path.join(output_dir, output_filename)

        try:
            (
                ffmpeg.input(audio_file_path, ss=start_time)
                .output(output_file, t=duration, **{"b:a": "128k"})
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            print(f"Saved segment {i + 1}/{len(segments)} to {output_file}")
        except ffmpeg.Error as e:
            print(f"Error extracting segment {i}: {e.stderr.decode('utf-8')}")


def process_file(audio_file_path, language=None):
    """Process a single audio file with WhisperX."""
    # Acquire a lock for this file
    lock_path = get_lock_path(audio_file_path)
    lock_acquired, lock_file = acquire_lock(lock_path)

    if not lock_acquired:
        print(
            f"File {audio_file_path} is being processed by another instance. Skipping."
        )
        return None

    try:
        print(f"\nProcessing: {audio_file_path}")

        # Create output directory for this file
        relative_path = os.path.relpath(
            os.path.dirname(audio_file_path), PROCESSING_DIR
        )
        base_filename = os.path.basename(os.path.splitext(audio_file_path)[0])
        output_subdir = os.path.join(OUTPUT_DIR, relative_path, base_filename)
        os.makedirs(output_subdir, exist_ok=True)

        # Save JSON path (in same folder as input file)
        json_output_path = os.path.splitext(audio_file_path)[0] + ".json"
        with torch.inference_mode():
            asr_options = {
                "initial_prompt": INITIAL_PROMPT,
                "temperatures": TEMPERATURES,
                "beam_size": BEAM_SIZE,
                "best_of": BEST_OF,
            }

            vad_options = {"vad_onset": VAD_ONSET, "vad_offset": VAD_OFFSET}

            # Perform language detection if needed
            audio_duration = get_audio_duration(audio_file_path)

            if (
                language is None
                and LANGUAGE_DETECTION_MIN_PROB > 0
                and audio_duration > 30000
            ):
                segments_duration_ms = 30000

                language_detection_max_tries = min(
                    LANGUAGE_DETECTION_MAX_TRIES,
                    math.floor(audio_duration / segments_duration_ms),
                )

                segments_starts = distribute_segments_equally(
                    audio_duration, segments_duration_ms, language_detection_max_tries
                )

                print(
                    "Detecting languages on segments starting at "
                    + ", ".join(map(str, segments_starts))
                )

                detected_language_details = detect_language(
                    audio_file_path,
                    segments_starts,
                    LANGUAGE_DETECTION_MIN_PROB,
                    LANGUAGE_DETECTION_MAX_TRIES,
                    asr_options,
                    vad_options,
                )

                detected_language_code = detected_language_details["language"]
                detected_language_prob = detected_language_details["probability"]
                detected_language_iterations = detected_language_details["iterations"]

                print(
                    f"Detected language {detected_language_code} ({detected_language_prob:.2f}) after "
                    f"{detected_language_iterations} iterations."
                )

                language = detected_language_details["language"]

            # Load model and transcribe
            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(
                WHISPER_ARCH,
                DEVICE,
                compute_type=COMPUTE_TYPE,
                language=language,
                asr_options=asr_options,
                vad_options=vad_options,
            )

            if DEBUG:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6
            audio = whisperx.load_audio(audio_file_path)

            if DEBUG:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6
            result = model.transcribe(audio, batch_size=BATCH_SIZE)
            detected_language = result["language"]

            if DEBUG:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            # Align if needed
            if ALIGN_OUTPUT:
                if (
                    detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH
                    or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF
                ):
                    result = align(audio, result, DEBUG)
                else:
                    print(
                        f"Cannot align output as language {detected_language} is not supported for alignment"
                    )

            # Diarize if needed
            if DIARIZATION:
                result = diarize(
                    audio,
                    result,
                    DEBUG,
                    HUGGINGFACE_ACCESS_TOKEN,
                    MIN_SPEAKERS,
                    MAX_SPEAKERS,
                )
        result["filename"] = audio_file_path
        result["base_filename"] = os.path.basename(audio_file_path)
        result["instance_id"] = INSTANCE_ID
        # Process speaker segments
        if "segments" in result:
            result["concatenated_speaker_samples"] = concat_speaker_samples(
                result["segments"]
            )
            extract_and_save_segments(
                audio_file_path, result["concatenated_speaker_samples"], output_subdir
            )

        # Save the complete JSON result
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved transcription JSON to {json_output_path}")

        if DEBUG:
            print(
                f"Max GPU memory allocated: {torch.cuda.max_memory_reserved() / (1024**3):.2f} GB"
            )

        return result
    finally:
        # Always release the lock, even if an exception occurs
        release_lock(lock_file, lock_path)


def main():
    """Main function to run the audio processing pipeline."""
    print(f"Starting processing with Instance ID: {INSTANCE_ID} on GPU: {args.gpu}")

    # Get unprocessed files
    unprocessed_files = search_unprocessed_files()
    print(f"Found {len(unprocessed_files)} unprocessed audio files.")

    if not unprocessed_files:
        print("No unprocessed files to process.")
        return

    # Get weighted sample
    sample = get_weighted_sample(unprocessed_files, SAMPLE_SIZE)

    # Process each file in the sample
    for i, file_path in enumerate(sample):
        print(f"\nProcessing file {i + 1}/{len(sample)}: {file_path}")
        try:
            process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()


# Run with: CUDA_VISIBLE_DEVICES=0 HF_HOME="/media/bodza/Audio_Dataset/hf_cache/" LD_LIBRARY_PATH="/home/bodza/Amphion/preprocessors/Emilia/.venv/lib/python3.9/site-packages/nvidia/cudnn/lib;/home/bodza/miniconda3/envs/parlertts/lib/python3.10/site-packages/nvidia/cudnn/lib/" python process_files.py --gpu 0 --instance-id 0 --sample-size 10

# CUDA_VISIBLE_DEVICES=1 HF_HOME="/media/bodza/Audio_Dataset/hf_cache/" LD_LIBRARY_PATH="/home/bodza/Amphion/preprocessors/Emilia/.venv/lib/python3.9/site-packages/nvidia/cudnn/lib;/home/bodza/miniconda3/envs/parlertts/lib/python3.10/site-packages/nvidia/cudnn/lib/" python process_files.py --gpu 1 --instance-id 1 --sample-size 10