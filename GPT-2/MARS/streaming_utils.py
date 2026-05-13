"""
Streaming data loading utilities for MARS
Adapted from nanochat streaming implementation
Supports multiple datasets: FineWeb-Edu, OpenWebText
"""
import os
import torch
from datasets import load_dataset
from collections import deque
import tiktoken

# Dataset configurations
DATASET_CONFIGS = {
    "karpathy/fineweb-edu-100b-shuffle": {
        "hf_name": "karpathy/fineweb-edu-100b-shuffle",
    },
    "Skylion007/openwebtext": {
        "hf_name": "Skylion007/openwebtext",
    },
}

# Streaming configuration (can be overridden)
STREAMING_CONFIG = {
    "timeout": 7200,  # 2 hours
    "max_retries": 10,
    "dataset_name": "Skylion007/openwebtext",  # default dataset
    "validation_fraction": 0.005,  # 0.5% for validation (similar to prepare.py's 0.0005 test_size)
}

def configure_streaming_session():
    """Configure HTTP session for robust streaming"""
    try:
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        from huggingface_hub import configure_http_session
        import requests

        session = requests.Session()
        retry_strategy = Retry(
            total=STREAMING_CONFIG["max_retries"],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        configure_http_session(session)
    except ImportError:
        print("Warning: Could not configure HTTP session retry strategy")
        pass

def streaming_document_generator(split, ddp_rank, ddp_world_size):
    """
    Generator for streaming documents from HF with proper train/val split.

    Supports multiple datasets configured in DATASET_CONFIGS.
    The dataset is selected via STREAMING_CONFIG["dataset_name"].

    Train/Val Split Strategy:
    - Most streaming datasets only have a 'train' split
    - We deterministically split documents using modulo arithmetic based on index
    - Each document index is checked: if (idx % val_modulo) == 0, it goes to validation
    - This ensures train and val are mutually exclusive and reproducible
    - Default: 0.5% validation (similar to prepare.py's 0.05% test_size)
    """
    configure_streaming_session()

    # Get dataset configuration
    dataset_key = STREAMING_CONFIG["dataset_name"]
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[dataset_key]
    hf_dataset_name = dataset_config["hf_name"]
    validation_fraction = STREAMING_CONFIG.get("validation_fraction", 0.005)

    # Calculate modulo for validation split
    # validation_fraction = 0.005 => val_modulo = 200 (1 in every 200 docs is validation)
    val_modulo = max(1, int(1.0 / validation_fraction))

    print(f"Loading streaming dataset: {hf_dataset_name}")
    print(f"Using split '{split}' with validation_fraction={validation_fraction:.1%} (1 in {val_modulo})")

    dataset = load_dataset(
        hf_dataset_name,
        split="train",  # Both datasets only have 'train' split
        streaming=True,
        download_mode="force_redownload"
    )

    batch = []
    num_filtered = 0
    num_included = 0

    for idx, example in enumerate(dataset):
        # DDP distribution: each rank gets different documents
        if idx % ddp_world_size != ddp_rank:
            continue

        # Deterministic train/val split based on document index
        is_validation = (idx % val_modulo) == 0

        if split == "val" and is_validation:
            # Include this document in validation
            batch.append(example['text'])
            num_included += 1
        elif split == "train" and not is_validation:
            # Include this document in training
            batch.append(example['text'])
            num_included += 1
        else:
            # Skip this document (it belongs to the other split)
            num_filtered += 1

        if len(batch) >= 1024:
            yield batch
            batch = []
            # Print progress occasionally
            if num_included % 10240 == 0 and num_included > 0:
                print(f"[{split}] Processed {num_included} documents (filtered {num_filtered})")

    if batch:  # Yield remaining
        yield batch

    if num_included > 0:
        print(f"[{split}] Finished. Processed {num_included} documents (filtered {num_filtered})")

class StreamingBatchGenerator:
    """Manages token buffer and batch generation for streaming"""
    def __init__(self, split, block_size, ddp_rank, ddp_world_size, device):
        self.split = split
        self.block_size = block_size
        self.device = device
        self.device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        # Initialize tokenizer
        self.enc = tiktoken.get_encoding("gpt2")

        # Token buffer (deque for efficient append/pop)
        self.token_buffer = deque()

        # Document generator
        self.doc_gen = streaming_document_generator(split, ddp_rank, ddp_world_size)

    def _refill_buffer(self, min_tokens_needed):
        """Refill token buffer from document stream"""
        while len(self.token_buffer) < min_tokens_needed:
            try:
                docs = next(self.doc_gen)
                for doc in docs:
                    tokens = self.enc.encode_ordinary(doc)
                    tokens.append(self.enc.eot_token)  # Add end-of-text token
                    self.token_buffer.extend(tokens)
            except StopIteration:
                # Restart generator if we run out (infinite loop)
                self.doc_gen = streaming_document_generator(
                    self.split,
                    self.ddp_rank,
                    self.ddp_world_size
                )

    def get_batch(self, batch_size):
        """Generate a batch from streaming data"""
        min_tokens = (self.block_size + 1) * batch_size
        self._refill_buffer(min_tokens)

        # Extract batch
        x_list = []
        y_list = []
        for _ in range(batch_size):
            # Get block_size + 1 tokens for input and target
            seq = [self.token_buffer.popleft() for _ in range(self.block_size + 1)]
            x_list.append(torch.tensor(seq[:-1], dtype=torch.long))
            y_list.append(torch.tensor(seq[1:], dtype=torch.long))

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        # Move to device
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y
