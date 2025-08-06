import torch
from typing import List, Tuple, Optional
import os
from pathlib import Path
from genslm import GenSLM, SequenceDataset
from Bio.Seq import Seq
import re
import intel_extension_for_pytorch as ipex
from accelerate import Accelerator

if torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu") # UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES) if run on xpu
    
accelerator = Accelerator()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def extract_id(header, id = 0):
    match = re.search(r'^([^|]+)', header)
    return match.group(id) if match else None


def build_prompt(dna, prompt_codon_len, model):
    dna = dna.upper()

    if len(dna) % 3 != 0:
        raise ValueError(f"DNA sequence length ({len(dna)}) is not divisible by 3")
    
    codons = [dna[i:i+3] for i in range(0, len(dna), 3)]
    prom_len = prompt_codon_len
    prom = codons[:prom_len]
    prompt = model.tokenizer.encode(' '.join(prom), return_tensors="pt").to(device)

    return prompt


def load_genslm_model(model_path: str, checkpoint_path: Optional[str] = None):
    print("🔁 Loading model...")
    model = GenSLM("genslm_2.5B_patric", model_cache_dir=model_path)
    tokenizer = model.tokenizer
    tokenizer.model_max_length = 501

    if checkpoint_path:
        ckpt_file = Path(checkpoint_path) / "pytorch_model_fsdp.bin"
        if ckpt_file.exists():
            print(f"📦 Loading checkpoint: {ckpt_file}")
            model.load_state_dict(torch.load(ckpt_file, map_location="cpu"), strict=False)
        elif Path(checkpoint_path).exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    else:
        print("⚠️ No checkpoint provided, using pretrained model weights.")

    model.model = model.model.to(device)
    model.eval()
    return model, tokenizer


def generate(
    raw_dna: str,
    model,
    tokenizer,
    prompt_codons: int = 2,
    gen_max_length: int = 390,
    gen_min_length: int = 330,
    num_outputs: int = 2,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> List[str]:
    
    prompt = build_prompt(raw_dna, prompt_codons, model)

    if gen_max_length <= prompt.shape[1]:
        raise ValueError(f"max_length={gen_max_length} ≤ prompt_len={prompt.shape[1]} — increase your generation length!")

    tokens = model.model.generate(
        prompt,
        max_length=gen_max_length,
        min_length=gen_min_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_outputs,
        remove_invalid_values=True,
        use_cache=True,
        pad_token_id=tokenizer.encode("[PAD]")[0],
        temperature=temperature,
    )

    generated_dna = tokenizer.batch_decode(tokens, skip_special_tokens=True)

    proteins = []
    for seq in generated_dna:
        dna_seq = seq.replace(" ", "")
        protein_seq = str(Seq(dna_seq).translate(to_stop=True))
        proteins.append(protein_seq)

    return proteins


if __name__ == "__main__":
    model_cache_dir="/lus/flare/projects/FoundEpidem/xlian/models/genslm_models/2.5B"
    checkpoint_dir = '/lus/flare/projects/FoundEpidem/xlian/verAB_genslm/genslm/runs_prod/nolog-4gpu-1ep-allbact/checkpoint-2929'

    model, tokenizer = load_genslm_model(model_path=model_cache_dir, checkpoint_path="path/to/checkpoint")
    protein_sequences = generate("ATGGCGTAA", model, tokenizer)
    
    for i in protein_sequences:
        print(i)