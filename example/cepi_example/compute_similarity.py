import re
import json
from Bio import SeqIO
from Levenshtein import ratio
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


generated_path = "output/generated.fasta"
original_path = "input/dna.fasta"


def extract_base_id(header):
    """Extracts the base ID (e.g., 'gdmA') from a header like 'gdmA|1'."""
    match = re.search(r'^([^|]+)', header)
    return match.group(1) if match else None

def extract_full_id(header):
    """Extracts the full ID (e.g., 'gdmA|1') from the header."""
    return header # Assuming the whole ID is what's needed. More complex regex could be used if needed.

# --- Load Original Sequences ---
# Load original DNA and translate them into protein sequences.
# The dictionary key will be the base ID (e.g., 'gdmA').
print("Loading and translating original DNA sequences...")
original_protein_records = {
    extract_base_id(r.id): str(r.seq.translate(stop_symbol="")) # Use stop_symbol="" to avoid '*'
    for r in SeqIO.parse(original_path, "fasta")
}

# --- Load and Group Generated Sequences ---
# Group generated sequences by their base ID.
# The result will be a dictionary like: {'gdmA': ['seq1', 'seq2', ...], 'vanA': [...]}
print("Loading and grouping generated protein sequences...")
generated_grouped_records = defaultdict(list)
for record in SeqIO.parse(generated_path, "fasta"):
    base_id = extract_base_id(record.id)
    if base_id:
        generated_grouped_records[base_id].append(str(record.seq))

# --- Main Comparison Loop ---
# This will be our final nested dictionary.
similarity_dict = {}

print("Calculating similarities...")
# Iterate through each base ID found in the original sequences.
for base_id, original_protein_seq in original_protein_records.items():
    
    # Check if this base ID has any generated sequences.
    if base_id not in generated_grouped_records:
        print(f"⚠️ No generated sequences found for base ID '{base_id}'. Skipping.")
        continue

    # Prepare a sub-dictionary to hold scores for the variants of this base ID.
    variant_scores = {}
    
    # Get the list of generated sequences for this base ID.
    generated_variants = generated_grouped_records[base_id]
    
    # Loop through each generated variant, keeping track of its index (1, 2, 3...).
    for i, generated_seq in enumerate(generated_variants):
        
        # Calculate the Levenshtein similarity ratio.
        similarity_score = ratio(generated_seq[100:], original_protein_seq[100:])
        
        # Store the score with the variant number (e.g., "1", "2", ...) as the key.
        variant_scores[str(i + 1)] = similarity_score
        
    # Add the completed variant scores dictionary to the main dictionary.
    similarity_dict[base_id] = variant_scores


'''
output_path = "output/uid_similarity.json"
with open(output_path, "w") as f:
    json.dump(similarity_dict, f, indent=4) # Use indent=4 for nice formatting

print(f"✅ DONE. Wrote results to {output_path}")
'''

# Convert to DataFrame
rows = []
for base_id, variants in similarity_dict.items():
    for variant_id, score in variants.items():
        rows.append((base_id, float(score)))
df = pd.DataFrame(rows, columns=["Gene", "Similarity"])

# Plotting
plt.figure(figsize=(5,5), dpi = 300)

# Create custom violin plot using matplotlib only
positions = []
data = []
xticks = []
colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"]
for i, gene in enumerate(sorted(df["Gene"].unique())):
    group_data = df[df["Gene"] == gene]["Similarity"].values
    parts = plt.violinplot(group_data, positions=[i], showmeans=False, showextrema=False, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)
    # Also plot individual points
    plt.scatter([i] * len(group_data), group_data, color="black", s=10, zorder=3)
    positions.append(i)
    xticks.append(gene)

plt.xticks(positions, xticks)
plt.ylabel("Levenshtein Similarity")
plt.title("Protein Similarity Scores per Gene\n(Generaed region only)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('fig/similarity.png')