import mne
import numpy as np
from scipy import signal
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt

# ── Custom Dataset ────────────────────────────────────────────────────────────
class EEGTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        with open(file_path, 'r') as f:
            text = f.read()
        tokenized = tokenizer(text, return_tensors='pt', truncation=False)
        self.examples = []
        input_ids = tokenized['input_ids'][0]
        for i in range(0, len(input_ids) - block_size, block_size):
            self.examples.append(input_ids[i:i+block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {'input_ids': self.examples[idx],
                'labels':    self.examples[idx]}

# ── Tokenizer functions ───────────────────────────────────────────────────────
sfreq = 256
window_size = int(2 * sfreq)
temporal_channels = [1, 2, 3]

def get_band_powers(window, sfreq=256):
    freqs, psd = signal.welch(window, fs=sfreq, nperseg=min(256, len(window)))
    bands = {'D':(0.5,4),'T':(4,8),'A':(8,13),'B':(13,30),'G':(30,40)}
    powers = {}
    total = 0
    for name,(lo,hi) in bands.items():
        idx = np.where((freqs>=lo)&(freqs<=hi))[0]
        powers[name] = np.sum(psd[idx])
        total += powers[name]
    rel = {k:v/total for k,v in powers.items()}
    return max(rel, key=rel.get)

def tokenize_recording(raw, channels=[1,2,3]):
    data = raw.get_data()
    n_windows = data.shape[1] // window_size
    tokens = []
    for i in range(n_windows):
        votes = []
        for ch in channels:
            window = data[ch, i*window_size:(i+1)*window_size]
            votes.append(get_band_powers(window))
        tokens.append(max(set(votes), key=votes.count))
    return tokens

def compute_perplexity(token_list, model, tokenizer, device='cpu'):
    model.eval()
    text = ' '.join(token_list)
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

# ── Step 1: Load and tokenize EEG ────────────────────────────────────────────
print("=== Step 1: Load and tokenize EEG ===")

print("Loading healthy EEG (file 1)...")
raw_h = mne.io.read_raw_edf('data/chbmit/chb01_01.edf', preload=True, verbose=False)
healthy_tokens = tokenize_recording(raw_h)
print(f"Healthy tokens: {len(healthy_tokens)}")

print("Loading seizure recording (file 2)...")
raw_s = mne.io.read_raw_edf('data/chbmit/chb01_02.edf', preload=True, verbose=False)
all_seizure_file_tokens = tokenize_recording(raw_s)

seizure_window_idx = 1467 // 2
pre_seizure_tokens = all_seizure_file_tokens[:700]
seizure_tokens     = all_seizure_file_tokens[seizure_window_idx : seizure_window_idx+15]
train_tokens       = healthy_tokens + pre_seizure_tokens

print(f"Training tokens (healthy only): {len(train_tokens)}")
print(f"Seizure tokens to test:         {len(seizure_tokens)}")
print(f"Seizure token sequence:         {''.join(seizure_tokens)}")

# ── Step 2: Save training text ────────────────────────────────────────────────
print("\n=== Step 2: Save training text ===")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

context_len = 60
lines = []
for i in range(0, len(train_tokens) - context_len, context_len):
    chunk = train_tokens[i:i+context_len]
    lines.append(' '.join(chunk))

with open('models/train_healthy.txt', 'w') as f:
    f.write('\n'.join(lines))

print(f"Training lines written: {len(lines)}")
print(f"Sample line: {lines[0]}")

# ── Step 3: Train GPT-2 ───────────────────────────────────────────────────────
print("\n=== Step 3: Train GPT-2 on healthy EEG tokens ===")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

dataset = EEGTextDataset(
    tokenizer=tokenizer,
    file_path='models/train_healthy.txt',
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir='models/gpt2_eeg',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    logging_steps=50,
    use_cpu=not torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("Training started — 5 to 15 minutes on CPU...")
trainer.train()
model.save_pretrained('models/gpt2_eeg')
tokenizer.save_pretrained('models/gpt2_eeg')
print("Model saved!")

# ── Step 4: Compute perplexity ────────────────────────────────────────────────
print("\n=== Step 4: Compute perplexity ===")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"Using device: {device}")

print("Computing perplexity on healthy windows...")
healthy_ppls = []
for i in range(0, min(100, len(healthy_tokens) - 60), 60):
    chunk = healthy_tokens[i:i+60]
    ppl = compute_perplexity(chunk, model, tokenizer, device)
    healthy_ppls.append(ppl)

print("Computing perplexity on pre-seizure window...")
pre_seizure_chunk = all_seizure_file_tokens[600:660]
pre_seizure_ppl   = compute_perplexity(pre_seizure_chunk, model, tokenizer, device)

print("Computing perplexity on seizure window...")
seizure_ppl = compute_perplexity(seizure_tokens, model, tokenizer, device)

# ── Step 5: Print results ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Mean healthy perplexity:   {np.mean(healthy_ppls):.2f}")
print(f"Std  healthy perplexity:   {np.std(healthy_ppls):.2f}")
print(f"Pre-seizure perplexity:    {pre_seizure_ppl:.2f}")
print(f"SEIZURE perplexity:        {seizure_ppl:.2f}")
print(f"Seizure / Healthy ratio:   {seizure_ppl / np.mean(healthy_ppls):.2f}x")

threshold = np.mean(healthy_ppls) + 2 * np.std(healthy_ppls)
print(f"Threshold (mean + 2*std):  {threshold:.2f}")
print("="*50)

if seizure_ppl > threshold:
    print("✅ HYPOTHESIS SUPPORTED: Seizure perplexity is significantly higher!")
    print("   Your core research idea is validated.")
else:
    print("⚠️  Seizure not clearly above threshold — check tokenization")

# ── Step 6: Plot results ──────────────────────────────────────────────────────
print("\n=== Step 5: Plotting results ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].boxplot(
    [healthy_ppls, [seizure_ppl]],
    labels=['Healthy EEG', 'Seizure EEG'],
    patch_artist=True,
    boxprops=dict(facecolor='lightblue')
)
axes[0].scatter([2], [seizure_ppl], color='red', s=150, zorder=5,
                label=f'Seizure ({seizure_ppl:.1f})')
axes[0].axhline(y=threshold, color='orange', linestyle='--',
                linewidth=1.5, label=f'Threshold ({threshold:.1f})')
axes[0].set_title('Perplexity Distribution: Healthy vs Seizure',
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('Perplexity (LLM surprise score)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

categories = ['Healthy\n(mean)', 'Pre-seizure', 'Seizure\n(ictal)']
values     = [np.mean(healthy_ppls), pre_seizure_ppl, seizure_ppl]
colors     = ['#2ecc71', '#f39c12', '#e74c3c']
bars = axes[1].bar(categories, values, color=colors,
                   alpha=0.85, edgecolor='black', width=0.5)
axes[1].axhline(y=threshold, color='orange', linestyle='--',
                linewidth=1.5, label=f'Threshold ({threshold:.1f})')
axes[1].set_title('Perplexity Comparison', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Perplexity')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Proof of Concept — LLM Perplexity as EEG Anomaly Detector',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/perplexity_poc.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved to results/perplexity_poc.png")
print("\n🎉 PoC complete!")