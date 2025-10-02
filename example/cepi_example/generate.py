from generate_util import *
from Bio import SeqIO
from tqdm import tqdm

vana_file = 'input/dna.fasta'
generate_path = 'output/generated.fasta'

#model_cache_dir="/lus/flare/projects/FoundEpidem/xlian/models/genslm_models/2.5B"
#checkpoint_dir = '/lus/flare/projects/FoundEpidem/xlian/verAB_genslm/genslm/runs_prod/nolog-4gpu-1ep-allbact/checkpoint-2929'
model_cache_dir="/home/yadunand/stash_dir/2.5B"
checkpoint_dir = '/home/yadunand/stash_dir/checkpoint-2929'

model, tokenizer = load_genslm_model(model_path=model_cache_dir, checkpoint_path=checkpoint_dir)

vana_records = list(SeqIO.parse(vana_file, "fasta"))
vanA_seqs = [(extract_id(r.id), str(r.seq)) for r in vana_records]

protein_sequences = []

for idx, (gene_name, dna_seq) in enumerate(tqdm(vanA_seqs, desc="🚀 Generating")):
    sample = generate(
                    dna_seq,
                    model,
                    tokenizer,
                    prompt_codons = 100,
                    gen_max_length = 375,
                    gen_min_length = 325,
                    num_outputs = 10,
                    )

    for i, seq in enumerate(sample, 1):
        header = f"{gene_name}|{i}"
        protein_sequences.append((header, seq))

with open(generate_path, "w") as f:
    for header, seq in protein_sequences:
        f.write(f">{header}\n{seq}\n")
