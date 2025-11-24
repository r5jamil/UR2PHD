import numpy as np
import torch
from typing import Iterator, List, Tuple
from transformers import GPT2Config
import time
import logging
from tqdm import tqdm
from glob import glob
from bgpt.utils import bGPTLMHeadModel
from bgpt.config import *
from arithmetic_coder import ac_utils, arithmetic_coder
from LLMCompress import write_padded_bytes, read_padded_bytes, Metric
import os
import sys
from bmp_utils import split_bmp_to_patches, merge_patches_to_bmp
import wave
import struct

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ==================== Configuration ====================
class CompressionConfig:
    """Configuration for bGPT compression"""
    # Model paths
    MODEL_CHECKPOINT_IMAGE = "./pretrained/bgpt/weights-image.pth"
    MODEL_CHECKPOINT_AUDIO = "./pretrained/bgpt/weights-audio.pth"
    
    # Dataset paths
    DATASET_IMAGE = "datasets/clic_2024/bmp/*.bmp"
    DATASET_AUDIO = "datasets/librispeech/wav/*.wav"
    TEST_DATASET_IMAGE = "datasets/test_workflow/bmp/*.bmp"
    TEST_DATASET_AUDIO = "datasets/test_workflow/wav/*.wav"
    
    # Output paths
    COMPRESSED_OUTPUT = "compressed.bin"
    
    # Model configuration
    PATCH_LENGTH = 512  # modify to fit the trained checkpoint
    PATCH_NUM_LAYERS = PATCH_NUM_LAYERS  # from bgpt.config
    BYTE_NUM_LAYERS = BYTE_NUM_LAYERS    # from bgpt.config
    HIDDEN_SIZE = HIDDEN_SIZE            # from bgpt.config
    PATCH_SIZE = PATCH_SIZE              # from bgpt.config
    
    # Compression parameters
    PRECISION = 64
    PREFIX_LENGTH = 1
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data type (for automatic path selection)
    DATA_TYPE = "audio"  # or "image"
    
    # BMP splitting parameters
    BMP_PATCH_SIZE = 32  # Size of square patches for BMP splitting
    
    # Audio splitting parameters
    AUDIO_CHUNK_DURATION = 1.0  # Duration of each audio chunk in seconds
    
    @classmethod
    def get_model_checkpoint(cls):
        """Get model checkpoint path based on data type"""
        return cls.MODEL_CHECKPOINT_IMAGE if cls.DATA_TYPE == "image" else cls.MODEL_CHECKPOINT_AUDIO
    
    @classmethod
    def get_dataset_path(cls, test: bool = False):
        """Get dataset path based on data type"""
        if test:
            return cls.TEST_DATASET_IMAGE if cls.DATA_TYPE == "image" else cls.TEST_DATASET_AUDIO
        else:
            return cls.DATASET_IMAGE if cls.DATA_TYPE == "image" else cls.DATASET_AUDIO

# ==================== Helper Functions ====================
def pad_input_for_bgpt(segments, ext_list, device, pad_to_length=None):
    """
    Pads input segments for bGPT model.
    Could be used for batch processing.
    
    :param segments: list of byte segments
    :param ext_list: list of extension bytes corresponding to each segment
    :param device: torch device
    :param pad_to_length: optional fixed padding length
    :return: dict with padded patches and masks
    """
    # 1. find longest
    max_length = max(len(b) for b in segments) + 2 * CompressionConfig.PATCH_SIZE

    padded_bytes = []
    padded_masks = []

    # 2. padding
    for b, ext in zip(segments, ext_list):
        if pad_to_length is not None and len(b) < pad_to_length:
            b = b + [256] * (pad_to_length - len(b))
        bos_patch = ext + [256] * (CompressionConfig.PATCH_SIZE - len(ext))
        b = bos_patch + b + [256] * CompressionConfig.PATCH_SIZE

        valid_length = len(b)
        padded_bytes.append(b + [256] * (max_length - valid_length))

        # Generate patch-level masks
        # Each patch contains PATCH_SIZE bytes, so we need (valid_length // PATCH_SIZE) masks
        patch_count = (valid_length + CompressionConfig.PATCH_SIZE - 1) // CompressionConfig.PATCH_SIZE  # Ceiling division
        total_patches = (
            max_length + CompressionConfig.PATCH_SIZE - 1
        ) // CompressionConfig.PATCH_SIZE  # Total number of patches after padding
        patch_masks = [1] * patch_count + [0] * (
            total_patches - patch_count
        )  # Active patches + padded patches
        padded_masks.append(patch_masks)

    patches = torch.tensor(padded_bytes, dtype=torch.long)
    masks = torch.tensor(padded_masks, dtype=torch.long)

    return {
        "patches": patches.to(device),
        "masks": masks.to(device),
    }


def bgpt_compress(compress_input, logits, metric, precision=None, prefix_length=None):
    """
    :param compress_input: symbols to be compressed
    :param logits: generation probabilities from the model
    :param metric: compression metrics
    :param precision: encoder precision
    :param prefix_length: prefix length for encoding
    :return: compressed result, a floating number
    """
    if precision is None:
        precision = CompressionConfig.PRECISION
    if prefix_length is None:
        prefix_length = CompressionConfig.PREFIX_LENGTH
        
    output = []
    # Initialize a Encoder Object
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=precision,
        output_fn=output.append,
    )
    # the first symbol should be saved for generation in decoding
    start_symbol = compress_input[:, :1]

    target_sequence_to_encode = compress_input[:, prefix_length:]
    logits_for_encoding = logits[:, prefix_length - 1 :, :]

    probs = logits_for_encoding.softmax(dim=-1).to(torch.float32)
    pd = torch.gather(
        probs, dim=-1, index=target_sequence_to_encode.unsqueeze(-1)
    ).squeeze(-1)

    probs = np.vstack(probs.detach().cpu().numpy().squeeze())

    sequence_array = target_sequence_to_encode.detach().cpu().numpy().reshape(-1)

    pd = pd.squeeze()

    # compress the sequence
    for symbol, prob, pd_prob in zip(sequence_array, probs, pd):
        encoder.encode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob, np.float32), symbol
        )
    encoder.terminate()

    # to visualize and compute metrics, map to str
    compressed_bits = "".join(map(str, output))
    # you can only save in bytes, so need to pad some bits
    compressed_bytes, num_padded_bits = ac_utils.bits_to_bytes(compressed_bits)
    
    metric.accumulate(len(compressed_bytes), len(sequence_array))

    compress_rate, compress_ratio = metric.compute_ratio()
    logger.info(f"compressed length: {metric.compressed_length}")
    logger.info(f"original length: {metric.total_length}")
    logger.info(f"compression ratio: {compress_ratio:.6f}")
    logger.info(f"compression rate: {compress_rate:.6f}")

    return compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs


def bgpt_decode(
    compressed_bytes,
    num_padded_bits,
    model,
    start_patch,
    ext,
    device,
    original_seq_len,
    original_sequence=None,
    pd=None,
    probs=None,
    precision=None,
    do_test=False,
):
    """
    :param compressed_bytes: compressed data
    :param num_padded_bits: padded bits
    :param model: same model as encoder
    :param start_patch: starting patch for decoding
    :param ext: file extension bytes
    :param device: torch device
    :param original_seq_len: original sequence length
    :param original_sequence: original symbol sequence, for testing purpose
    :param pd: actually not needed, used for testing
    :param probs: probabilities from encoder
    :param precision: decoder precision
    :param do_test: whether to run testing
    :return: decoded sequence
    """
    if precision is None:
        precision = CompressionConfig.PRECISION
        
    # convert bytes back to bit stream
    data_iter = iter(
        ac_utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits)
    )

    # utils function to read bits
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    # initialize a Decoder Object
    decoder = arithmetic_coder.Decoder(
        base=2,
        precision=precision,
        input_fn=_input_fn,
    )

    # loop for decompressing
    target_diff_list = []
    target_in_top5_list = []

    # start_symbol should be empty for bgpt
    start_symbol = []
    sequence_array_de = np.array(start_symbol)

    for i in range(original_seq_len):

        sequence_array_de = sequence_array_de[None, :].tolist()
        sequence_array_de_input = pad_input_for_bgpt(sequence_array_de, [ext], device, original_seq_len)
        
        logits = model(**sequence_array_de_input).logits
        logits = logits[:-1, :-1, :]
        prob_de = logits.reshape(1, -1, 257).softmax(-1).detach().cpu().numpy().squeeze(axis=0)

        de_token = decoder.decode(
            ac_utils.normalize_pdf_for_arithmetic_coding(prob_de[i], data_type=np.float32),
        )
        sequence_array_de = np.append(sequence_array_de, de_token)

        current_len = len(sequence_array_de)
        target_len = original_seq_len

        if current_len < target_len:
            padded = np.pad(
                sequence_array_de, (0, (target_len - current_len)), constant_values=0
            )
        else:
            padded = sequence_array_de
        sequence_array_de_input = torch.tensor(
            padded, dtype=torch.long, device=device
        ).unsqueeze(0)

        if do_test:
            top_indices_de = prob_de[i].argsort()[-5:][::-1]
            top_indices = probs[i].argsort()[-5:][::-1]

            # target diff
            target_diff = (
                probs[i, original_sequence[i]] - prob_de[i, original_sequence[i]]
            )
            target_diff_list.append(target_diff)

            # target in top 5
            target_in_top5 = original_sequence[i] in top_indices
            target_in_top5_list.append(target_in_top5)
            print(
                f"idx: {i}, original token: {original_sequence[i]}, decoder token: {de_token}"
            )
            print(
                f"diff probs max: {max(abs(probs[i] - prob_de[i]))}, original sum error: {abs(sum(prob_de[i]) - 1.0)}, decoder sum error: {abs(sum(probs[i]) - 1.0)}"
            )
            print(
                f"original: {top_indices}, target_in_top5: {target_in_top5} decode: {top_indices_de}, "
            )
            print(f"target diff: {target_diff}")
            if original_sequence[i] != de_token:
                import pdb
                pdb.set_trace()

    return sequence_array_de_input


def read_bytes(filename):
    """
    Read bytes from file and extract extension
    :param filename: path to file
    :return: tuple of (bytes list, extension bytes)
    """
    # ext should be 'bmp' or 'wav'
    ext = filename.split(".")[-1]

    ext = bytearray(ext, "utf-8")
    ext = [byte for byte in ext][:CompressionConfig.PATCH_SIZE]
    with open(filename, "rb") as f:
        file_bytes = f.read()

    bytes_list = []
    for byte in file_bytes:
        bytes_list.append(byte)

    if len(bytes_list) % CompressionConfig.PATCH_SIZE != 0:
        bytes_list = bytes_list + [256] * (CompressionConfig.PATCH_SIZE - len(bytes_list) % CompressionConfig.PATCH_SIZE)

    return bytes_list, ext


def write_bytes(filename, bytes_list):
    """
    Write bytes list to file
    :param filename: output file path
    :param bytes_list: list of bytes to write
    """
    # Remove padding (256 values)
    while bytes_list and bytes_list[-1] == 256:
        bytes_list = bytes_list[:-1]
    
    # Convert to bytes and write
    byte_array = bytearray(bytes_list)
    with open(filename, "wb") as f:
        f.write(byte_array)


def split_wav_to_chunks(wav_file, output_folder, chunk_duration=1.0):
    """
    Split a WAV file into chunks of specified duration.
    
    :param wav_file: path to input WAV file
    :param output_folder: folder to save chunks
    :param chunk_duration: duration of each chunk in seconds
    :return: tuple of (list of chunk file paths, WAV parameters dict)
    """
    # Read WAV file
    with wave.open(wav_file, 'rb') as wav:
        params = wav.getparams()
        n_channels = params.nchannels
        sampwidth = params.sampwidth
        framerate = params.framerate
        n_frames = params.nframes
        
        # Calculate chunk size in frames
        chunk_frames = int(framerate * chunk_duration)
        
        # Read all frames
        frames = wav.readframes(n_frames)
    
    # Create output folder
    filename = os.path.basename(wav_file)
    name_without_ext = os.path.splitext(filename)[0]
    chunk_folder = os.path.join(output_folder, name_without_ext)
    os.makedirs(chunk_folder, exist_ok=True)
    
    # Split into chunks
    chunk_files = []
    chunk_idx = 0
    
    bytes_per_frame = n_channels * sampwidth
    total_bytes = len(frames)
    chunk_bytes = chunk_frames * bytes_per_frame
    
    for start_byte in range(0, total_bytes, chunk_bytes):
        end_byte = min(start_byte + chunk_bytes, total_bytes)
        chunk_data = frames[start_byte:end_byte]
        
        # Save chunk
        chunk_filename = f"chunk_{chunk_idx:04d}.wav"
        chunk_path = os.path.join(chunk_folder, chunk_filename)
        
        with wave.open(chunk_path, 'wb') as chunk_wav:
            chunk_wav.setparams(params)
            chunk_wav.writeframes(chunk_data)
        
        chunk_files.append(chunk_path)
        chunk_idx += 1
    
    # Store WAV parameters for reconstruction
    wav_params = {
        'nchannels': n_channels,
        'sampwidth': sampwidth,
        'framerate': framerate,
        'comptype': params.comptype,
        'compname': params.compname,
    }
    
    return chunk_files, wav_params


def merge_wav_chunks(chunk_files, output_path, wav_params):
    """
    Merge WAV chunks back into a single WAV file.
    
    :param chunk_files: list of chunk file paths (in order)
    :param output_path: path for output merged WAV file
    :param wav_params: WAV parameters dict from split_wav_to_chunks
    """
    # Read all chunks
    all_frames = []
    
    for chunk_file in sorted(chunk_files):
        with wave.open(chunk_file, 'rb') as chunk_wav:
            frames = chunk_wav.readframes(chunk_wav.getnframes())
            all_frames.append(frames)
    
    # Merge and write
    with wave.open(output_path, 'wb') as output_wav:
        output_wav.setnchannels(wav_params['nchannels'])
        output_wav.setsampwidth(wav_params['sampwidth'])
        output_wav.setframerate(wav_params['framerate'])
        output_wav.setcomptype(wav_params['comptype'], wav_params['compname'])
        
        for frames in all_frames:
            output_wav.writeframes(frames)


def load_bgpt_model(checkpoint_path, device):
    """
    Load bGPT model from checkpoint
    :param checkpoint_path: path to model checkpoint
    :param device: torch device
    :return: loaded model
    """
    print("Loading bGPT model...")
    
    patch_config = GPT2Config(
        num_hidden_layers=CompressionConfig.PATCH_NUM_LAYERS,
        max_length=CompressionConfig.PATCH_LENGTH,
        max_position_embeddings=CompressionConfig.PATCH_LENGTH,
        hidden_size=CompressionConfig.HIDDEN_SIZE,
        n_head=CompressionConfig.HIDDEN_SIZE // 64,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=CompressionConfig.BYTE_NUM_LAYERS,
        max_length=CompressionConfig.PATCH_SIZE + 1,
        max_position_embeddings=CompressionConfig.PATCH_SIZE + 1,
        hidden_size=CompressionConfig.HIDDEN_SIZE,
        n_head=CompressionConfig.HIDDEN_SIZE // 64,
        vocab_size=256 + 1,
    )
    llm = bGPTLMHeadModel(patch_config, byte_config)

    checkpoint = torch.load(checkpoint_path)
    # use this strict=False to tolerate transformers package version mismatch
    llm.load_state_dict(checkpoint["model"], strict=False)
    llm = llm.to(device)
    # llm = llm.to(torch.float16)
    llm.eval()

    print("Loaded bGPT model.")
    return llm


def load_dataset(dataset_path, device) -> List[Tuple[dict, List[int]]]:
    """
    Load dataset for compression
    :param dataset_path: glob pattern for dataset files
    :param device: torch device
    :return: list of tuples (padded_segment, ext)
    """
    print("Loading dataset for compression testing...")

    fs = glob(dataset_path)
    dataset = []
    
    for _, af in tqdm(enumerate(fs), total=len(fs)):
        bytes_list, ext = read_bytes(af)
        # Pad the segment and keep ext for later use
        padded_segment = pad_input_for_bgpt([bytes_list], [ext], device)
        dataset.append((padded_segment, ext))

    print(f"Loaded {len(dataset)} files for compression testing.")
    return dataset


def test_workflow(model, dataset, device, output_path):
    """
    Run compression and decompression workflow
    :param model: bGPT model
    :param dataset: list of tuples (padded_segment, ext)
    :param device: torch device
    :param output_path: path to save compressed output
    """
    compression_start_time = time.time()

    for segment, ext in dataset:

        metric = Metric()
        with torch.inference_mode():
            attention_mask = segment["masks"]
            input_ids = segment["patches"]
            output = model(patches=input_ids, masks=attention_mask)
            logits = output.logits

            # e.g.: logits: (511, PATCH_SIZE+1, 257)
            # Remove the last time step for each patch
            # Remove the prediction for the ending patch
            logits = logits[:-1, :-1, :]
            logits = logits.reshape(1, -1, 257)  # Flatten to (1, 510 * PATCH_SIZE, 257)

            # Adjust input_ids: Remove the first and last <PATCH_SIZE> tokens
            start_patch = input_ids[:, :CompressionConfig.PATCH_SIZE].squeeze(0)  # (PATCH_SIZE)
            input_ids = input_ids[:, CompressionConfig.PATCH_SIZE:-CompressionConfig.PATCH_SIZE]  # (1, 510 * PATCH_SIZE)
            # add just one meaningless token in the beginning for start symbol
            # to make bpgt fit in the arithmetic coding framework
            input_ids = torch.cat(
                [torch.tensor([[256]], device=device), input_ids], dim=1
            )

            # Adjust attention_mask
            attention_mask = attention_mask.repeat_interleave(CompressionConfig.PATCH_SIZE, dim=1)
            attention_mask = attention_mask[
                :, CompressionConfig.PATCH_SIZE:-CompressionConfig.PATCH_SIZE
            ]  # Align with input_ids

        compressed_bytes, num_padded_bits, _, sequence_array, pd, probs = (
            bgpt_compress(input_ids, logits, metric=metric)
        )

        compression_end_time = time.time()

        print("compressed_bytes:", compressed_bytes)
        print("num_padded_bits:", num_padded_bits)
        original_length = input_ids.shape[1] - 1  # exclude the meaningless starting token
        print("original_length:", original_length)
        write_padded_bytes(
            output_path, compressed_bytes, num_padded_bits, original_length
        )
        print(f"Wrote compressed data to {output_path}")
        print("Compression ratio/rate:", metric.compute_ratio())

        compressed_bytes, num_padded_bits, original_length = read_padded_bytes(
            output_path
        )
        print(f"Read compressed data from {output_path}")

        decompression_start_time = time.time()

        decompressed = bgpt_decode(
            compressed_bytes,
            num_padded_bits,
            model,
            start_patch,
            ext,  # Pass ext to decode function
            device,
            original_length,
            sequence_array,
            pd,
            probs,
            do_test=True,
        )

        decompression_end_time = time.time()

        print(
            f"Compression time: {compression_end_time - compression_start_time:.2f} seconds"
        )
        print(
            f"Decompression time: {decompression_end_time - decompression_start_time:.2f} seconds"
        )


def test_bmp_compression(
    model,
    device,
    test: bool = False,
    temp_folder: str = "temp",
    output_folder: str = "output",
    patch_size: int = None,
):
    """
    Test BMP file compression and decompression workflow.
    
    This function:
    1. Splits each BMP file into small patches
    2. Compresses each patch separately
    3. Decompresses each patch
    4. Merges patches back into the original image
    5. Verifies the reconstructed image matches the original
    
    :param model: bGPT model for compression
    :param device: torch device
    :param test: whether to use test dataset (from CompressionConfig)
    :param temp_folder: temporary folder for intermediate files
    :param output_folder: folder for final output
    :param patch_size: size of square patches for splitting BMP (default from config)
    """
    if patch_size is None:
        patch_size = CompressionConfig.BMP_PATCH_SIZE
    
    # Get dataset path from config
    dataset_path = CompressionConfig.get_dataset_path(test=test)
    
    # Find all BMP files
    bmp_files = glob(dataset_path)
    
    if not bmp_files:
        print(f"No BMP files found in {dataset_path}")
        return
    
    print(f"Found {len(bmp_files)} BMP files to test")
    print(f"Dataset path: {dataset_path}")
    print("=" * 80)
    
    # Create necessary folders
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    split_folder = os.path.join(temp_folder, "split")
    compressed_folder = os.path.join(temp_folder, "compressed")
    decompressed_folder = os.path.join(temp_folder, "decompressed")
    
    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(compressed_folder, exist_ok=True)
    os.makedirs(decompressed_folder, exist_ok=True)
    
    total_metric = Metric()
    
    for bmp_file in bmp_files:
        print(f"\nProcessing: {os.path.basename(bmp_file)}")
        print("-" * 80)
        
        filename = os.path.basename(bmp_file)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Step 1: Split BMP into patches
        print(f"Step 1: Splitting BMP into {patch_size}x{patch_size} patches...")
        split_bmp_to_patches(
            source_folder=os.path.dirname(bmp_file),
            output_folder=split_folder,
            patch_size=patch_size
        )
        
        patches_subfolder = os.path.join(split_folder, name_without_ext)
        patch_files = sorted(glob(os.path.join(patches_subfolder, "*.bmp")))
        print(f"Created {len(patch_files)} patches")
        
        # Step 2: Compress each patch
        print(f"\nStep 2: Compressing {len(patch_files)} patches...")
        compressed_info = {}  # Store compression info for each patch
        
        for patch_file in tqdm(patch_files, desc="Compressing patches"):
            patch_name = os.path.basename(patch_file)
            patch_id = os.path.splitext(patch_name)[0]
            
            # Read patch bytes
            bytes_list, ext = read_bytes(patch_file)
            
            # Prepare input
            padded_segment = pad_input_for_bgpt([bytes_list], [ext], device)
            
            metric = Metric()
            
            with torch.inference_mode():
                attention_mask = padded_segment["masks"]
                input_ids = padded_segment["patches"]
                output = model(patches=input_ids, masks=attention_mask)
                logits = output.logits
                
                logits = logits[:-1, :-1, :]
                logits = logits.reshape(1, -1, 257)
                
                start_patch = input_ids[:, :CompressionConfig.PATCH_SIZE].squeeze(0)
                input_ids = input_ids[:, CompressionConfig.PATCH_SIZE:-CompressionConfig.PATCH_SIZE]
                input_ids = torch.cat(
                    [torch.tensor([[256]], device=device), input_ids], dim=1
                )
            
            # Compress
            compressed_bytes, num_padded_bits, _, sequence_array, pd, probs = (
                bgpt_compress(input_ids, logits, metric=metric)
            )
            
            # Save compressed data
            compressed_path = os.path.join(compressed_folder, f"{patch_id}.bin")
            original_length = input_ids.shape[1] - 1
            write_padded_bytes(compressed_path, compressed_bytes, num_padded_bits, original_length)
            
            # Store info for decompression
            compressed_info[patch_id] = {
                'compressed_path': compressed_path,
                'start_patch': start_patch,
                'ext': ext,
                'original_length': original_length,
            }
            
            total_metric.accumulate(metric.compressed_length, metric.total_length)
        
        compress_rate, compress_ratio = total_metric.compute_ratio()
        print("Compression ratio/rate:", total_metric.compute_ratio())
        
        # Step 3: Decompress each patch
        print(f"\nStep 3: Decompressing {len(patch_files)} patches...")
        
        decompressed_subfolder = os.path.join(decompressed_folder, name_without_ext)
        os.makedirs(decompressed_subfolder, exist_ok=True)
        
        for patch_id, info in tqdm(compressed_info.items(), desc="Decompressing patches"):
            # Read compressed data
            compressed_bytes, num_padded_bits, original_length = read_padded_bytes(
                info['compressed_path']
            )
            
            # Decompress
            decompressed_tensor = bgpt_decode(
                compressed_bytes,
                num_padded_bits,
                model,
                info['start_patch'],
                info['ext'],
                device,
                original_length,
                do_test=False,
            )
            
            # Convert tensor to bytes and save
            decompressed_bytes = decompressed_tensor.squeeze(0).cpu().numpy().tolist()
            decompressed_path = os.path.join(decompressed_subfolder, f"{patch_id}.bmp")
            write_bytes(decompressed_path, decompressed_bytes)
        
        # Step 4: Merge patches back to original image
        print(f"\nStep 4: Merging patches back to original image...")
        reconstructed_path = os.path.join(output_folder, f"reconstructed_{filename}")
        merge_patches_to_bmp(
            patches_folder=decompressed_subfolder,
            output_path=reconstructed_path,
            patch_size=patch_size
        )
        
        # Step 5: Verify reconstruction
        print(f"\nStep 5: Verifying reconstruction...")
        original_bytes, _ = read_bytes(bmp_file)
        reconstructed_bytes, _ = read_bytes(reconstructed_path)
        
        # Remove padding from both
        while original_bytes and original_bytes[-1] == 256:
            original_bytes = original_bytes[:-1]
        while reconstructed_bytes and reconstructed_bytes[-1] == 256:
            reconstructed_bytes = reconstructed_bytes[:-1]
        
        if original_bytes == reconstructed_bytes:
            print(f"✓ Reconstruction successful! Files match perfectly.")
        else:
            print(f"✗ Warning: Reconstructed file differs from original")
            print(f"  Original size: {len(original_bytes)} bytes")
            print(f"  Reconstructed size: {len(reconstructed_bytes)} bytes")
            
            # Find first difference
            min_len = min(len(original_bytes), len(reconstructed_bytes))
            for i in range(min_len):
                if original_bytes[i] != reconstructed_bytes[i]:
                    print(f"  First difference at byte {i}: {original_bytes[i]} vs {reconstructed_bytes[i]}")
                    break
        
        print("=" * 80)
    
    print(f"\nBMP compression test completed!")
    print(f"Reconstructed images saved to: {output_folder}")
    print("Compression ratio/rate:", total_metric.compute_ratio())


def test_audio_compression(
    model,
    device,
    test: bool = False,
    temp_folder: str = "temp_audio",
    output_folder: str = "output_audio",
    chunk_duration: float = None,
):
    """
    Test audio file (WAV) compression and decompression workflow.
    
    This function:
    1. Splits each WAV file into chunks (e.g., 1 second each)
    2. Compresses each chunk separately
    3. Decompresses each chunk
    4. Merges chunks back into the original audio
    5. Verifies the reconstructed audio matches the original
    
    :param model: bGPT model for compression
    :param device: torch device
    :param test: whether to use test dataset (from CompressionConfig)
    :param temp_folder: temporary folder for intermediate files
    :param output_folder: folder for final output
    :param chunk_duration: duration of each audio chunk in seconds (default from config)
    """
    if chunk_duration is None:
        chunk_duration = CompressionConfig.AUDIO_CHUNK_DURATION
    
    # Temporarily switch to audio mode to get correct paths
    original_data_type = CompressionConfig.DATA_TYPE
    CompressionConfig.DATA_TYPE = "audio"
    
    # Get dataset path from config
    dataset_path = CompressionConfig.get_dataset_path(test=test)
    
    # Find all WAV files
    wav_files = glob(dataset_path)
    
    if not wav_files:
        print(f"No WAV files found in {dataset_path}")
        CompressionConfig.DATA_TYPE = original_data_type  # Restore
        return
    
    print(f"Found {len(wav_files)} WAV files to test")
    print(f"Dataset path: {dataset_path}")
    print(f"Chunk duration: {chunk_duration} seconds")
    print("=" * 80)
    
    # Create necessary folders
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    split_folder = os.path.join(temp_folder, "split")
    compressed_folder = os.path.join(temp_folder, "compressed")
    decompressed_folder = os.path.join(temp_folder, "decompressed")
    
    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(compressed_folder, exist_ok=True)
    os.makedirs(decompressed_folder, exist_ok=True)
    
    total_metric = Metric()
    
    for wav_file in wav_files:
        print(f"\nProcessing: {os.path.basename(wav_file)}")
        print("-" * 80)
        
        filename = os.path.basename(wav_file)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Step 1: Split WAV into chunks
        print(f"Step 1: Splitting WAV into {chunk_duration}s chunks...")
        chunk_files, wav_params = split_wav_to_chunks(
            wav_file,
            split_folder,
            chunk_duration=chunk_duration
        )
        print(f"Created {len(chunk_files)} chunks")
        print(f"WAV parameters: {wav_params['nchannels']} channels, "
              f"{wav_params['sampwidth']} bytes/sample, "
              f"{wav_params['framerate']} Hz")
        
        # Step 2: Compress each chunk
        print(f"\nStep 2: Compressing {len(chunk_files)} chunks...")
        compressed_info = {}  # Store compression info for each chunk
        
        for chunk_file in tqdm(chunk_files, desc="Compressing chunks"):
            chunk_name = os.path.basename(chunk_file)
            chunk_id = os.path.splitext(chunk_name)[0]
            
            # Read chunk bytes
            bytes_list, ext = read_bytes(chunk_file)
            
            # Prepare input
            padded_segment = pad_input_for_bgpt([bytes_list], [ext], device)
            
            metric = Metric()
            
            with torch.inference_mode():
                attention_mask = padded_segment["masks"]
                input_ids = padded_segment["patches"]
                output = model(patches=input_ids, masks=attention_mask)
                logits = output.logits
                
                # Process logits
                logits = logits[:-1, :-1, :]
                logits = logits.reshape(1, -1, 257)
                
                # Prepare input_ids
                start_patch = input_ids[:, :CompressionConfig.PATCH_SIZE].squeeze(0)
                input_ids = input_ids[:, CompressionConfig.PATCH_SIZE:-CompressionConfig.PATCH_SIZE]
                input_ids = torch.cat(
                    [torch.tensor([[256]], device=device), input_ids], dim=1
                )
            
            # Compress
            compressed_bytes, num_padded_bits, _, sequence_array, pd, probs = (
                bgpt_compress(input_ids, logits, metric=metric)
            )
            
            # Save compressed data
            compressed_path = os.path.join(compressed_folder, f"{name_without_ext}_{chunk_id}.bin")
            original_length = input_ids.shape[1] - 1
            write_padded_bytes(compressed_path, compressed_bytes, num_padded_bits, original_length)
            
            # Store info for decompression
            compressed_info[chunk_id] = {
                'compressed_path': compressed_path,
                'start_patch': start_patch,
                'ext': ext,
                'original_length': original_length,
            }
            
            total_metric.accumulate(metric.compressed_length, metric.total_length)
        
        print("Compression ratop/rate:", total_metric.compute_ratio())
        
        # Step 3: Decompress each chunk
        print(f"\nStep 3: Decompressing {len(chunk_files)} chunks...")
        
        decompressed_subfolder = os.path.join(decompressed_folder, name_without_ext)
        os.makedirs(decompressed_subfolder, exist_ok=True)
        
        decompressed_chunk_files = []
        
        for chunk_id in tqdm(sorted(compressed_info.keys()), desc="Decompressing chunks"):
            info = compressed_info[chunk_id]
            
            # Read compressed data
            compressed_bytes, num_padded_bits, original_length = read_padded_bytes(
                info['compressed_path']
            )
            
            # Decompress
            decompressed_tensor = bgpt_decode(
                compressed_bytes,
                num_padded_bits,
                model,
                info['start_patch'],
                info['ext'],
                device,
                original_length,
                do_test=False,
            )
            
            # Convert tensor to bytes and save
            decompressed_bytes = decompressed_tensor.squeeze(0).cpu().numpy().tolist()
            decompressed_path = os.path.join(decompressed_subfolder, f"{chunk_id}.wav")
            write_bytes(decompressed_path, decompressed_bytes)
            decompressed_chunk_files.append(decompressed_path)
        
        # Step 4: Merge chunks back to original audio
        print(f"\nStep 4: Merging chunks back to original audio...")
        reconstructed_path = os.path.join(output_folder, f"reconstructed_{filename}")
        merge_wav_chunks(
            decompressed_chunk_files,
            reconstructed_path,
            wav_params
        )
        
        # Step 5: Verify reconstruction
        print(f"\nStep 5: Verifying reconstruction...")
        original_bytes, _ = read_bytes(wav_file)
        reconstructed_bytes, _ = read_bytes(reconstructed_path)
        
        # Remove padding from both
        while original_bytes and original_bytes[-1] == 256:
            original_bytes = original_bytes[:-1]
        while reconstructed_bytes and reconstructed_bytes[-1] == 256:
            reconstructed_bytes = reconstructed_bytes[:-1]
        
        if original_bytes == reconstructed_bytes:
            print(f"✓ Reconstruction successful! Files match perfectly.")
            print(f"  File size: {len(original_bytes)} bytes")
        else:
            print(f"✗ Warning: Reconstructed file differs from original")
            print(f"  Original size: {len(original_bytes)} bytes")
            print(f"  Reconstructed size: {len(reconstructed_bytes)} bytes")
            
            # Find first difference
            min_len = min(len(original_bytes), len(reconstructed_bytes))
            for i in range(min_len):
                if original_bytes[i] != reconstructed_bytes[i]:
                    print(f"  First difference at byte {i}: {original_bytes[i]} vs {reconstructed_bytes[i]}")
                    break
        
        print("=" * 80)
    
    # Restore original data type
    CompressionConfig.DATA_TYPE = original_data_type
    
    # Print overall statistics
    print(f"\nAudio compression test completed!")
    print(f"Total files processed: {len(wav_files)}")
    print(f"Reconstructed audio files saved to: {output_folder}")
    print("Compression ratio/rate:", total_metric.compute_ratio())


if __name__ == "__main__":
    # Setup device
    device = torch.device(CompressionConfig.DEVICE)
    
    # Load model
    model_checkpoint = CompressionConfig.get_model_checkpoint()
    llm = load_bgpt_model(model_checkpoint, device)
    
    """
    # Option 1: Run standard workflow test
    print("\n" + "=" * 80)
    print("Running Standard Workflow Test")
    print("=" * 80)
    dataset_path = CompressionConfig.get_dataset_path(test=True)
    dataset = load_dataset(dataset_path, device)
    test_workflow(llm, dataset, device, CompressionConfig.COMPRESSED_OUTPUT)
    """

    """
    # Option 2: Run BMP compression test
    print("\n" + "=" * 80)
    print("Running BMP Compression Test")
    print("=" * 80)
    test_bmp_compression(
        model=llm,
        device=device,
        test=True,  # Set to True to use TEST_DATASET_IMAGE
        temp_folder="temp_img",
        output_folder="output_img",
        patch_size=CompressionConfig.BMP_PATCH_SIZE
    )
    """

    # Option 3: Run Audio compression test with chunking
    print("\n" + "=" * 80)
    print("Running Audio Compression Test (with chunking)")
    print("=" * 80)
    test_audio_compression(
        model=llm,
        device=device,
        test=True,  # Set to True to use TEST_DATASET_AUDIO
        temp_folder="temp_audio",
        output_folder="output_audio",
        chunk_duration=CompressionConfig.AUDIO_CHUNK_DURATION,  # 1.0 second chunks
    )