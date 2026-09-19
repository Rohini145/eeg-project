import mne
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

raw_healthy = mne.io.read_raw_edf('data/chbmit/chb01_01.edf', preload=True, verbose=False)
raw_seizure = mne.io.read_raw_edf('data/chbmit/chb01_02.edf', preload=True, verbose=False)

sfreq = 256
window_size = int(2 * sfreq)  # 2 seconds instead of 4

def get_band_powers(window, sfreq=256):
    """Returns power in ALL bands, not just the dominant one."""
    freqs, psd = signal.welch(window, fs=sfreq, nperseg=min(256, len(window)))
    bands = {
        'D': (0.5, 4),
        'T': (4, 8),
        'A': (8, 13),
        'B': (13, 30),
        'G': (30, 40),
    }
    powers = {}
    total = 0
    for name, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        powers[name] = np.sum(psd[idx])
        total += powers[name]
    
    # Relative power (normalized)
    rel_powers = {k: v/total for k, v in powers.items()}
    dominant = max(rel_powers, key=rel_powers.get)
    return dominant, rel_powers

def tokenize_multichannel(raw, label="", channels=[1, 2, 3]):
    """Use multiple temporal channels, take majority vote per window."""
    data = raw.get_data()
    n_windows = data.shape[1] // window_size
    
    tokens = []
    all_powers = {b: [] for b in 'DTABG'}
    
    for i in range(n_windows):
        votes = []
        window_powers = {b: 0 for b in 'DTABG'}
        
        for ch_idx in channels:
            window = data[ch_idx, i*window_size:(i+1)*window_size]
            dominant, powers = get_band_powers(window)
            votes.append(dominant)
            for b in 'DTABG':
                window_powers[b] += powers[b]
        
        # Majority vote across channels
        token = max(set(votes), key=votes.count)
        tokens.append(token)
        
        for b in 'DTABG':
            all_powers[b].append(window_powers[b] / len(channels))
    
    print(f"\n{label}")
    print(f"  Windows: {n_windows}")
    print(f"  First 60 tokens: {''.join(tokens[:60])}")
    print(f"  Token counts: { {t: tokens.count(t) for t in 'DTABG'} }")
    
    return tokens, all_powers

# Temporal channels: F7-T7(1), T7-P7(2), P7-O1(3) — best for seizure
temporal_channels = [1, 2, 3]

healthy_tokens, healthy_powers = tokenize_multichannel(
    raw_healthy, "HEALTHY EEG", temporal_channels)

seizure_tokens, seizure_powers = tokenize_multichannel(
    raw_seizure, "FULL SEIZURE RECORDING", temporal_channels)

# Now look specifically at seizure window vs healthy window
sfreq_w = 256
seizure_win_start = 1467 // 2
seizure_win_end   = 1497 // 2
healthy_win_start = 1000 // 2
healthy_win_end   = 1030 // 2

data_s = raw_seizure.get_data()
data_h = raw_healthy.get_data()

print("\n--- SEIZURE WINDOW BAND POWERS ---")
for ch_idx in temporal_channels:
    ch_name = raw_seizure.ch_names[ch_idx]
    
    # Healthy window from file 1
    h_win = data_h[ch_idx, healthy_win_start*window_size : healthy_win_end*window_size]
    _, h_pow = get_band_powers(h_win[:window_size])
    
    # Seizure window from file 2
    s_win = data_s[ch_idx, seizure_win_start*window_size : seizure_win_end*window_size]
    _, s_pow = get_band_powers(s_win[:window_size])
    
    print(f"\n  Channel {ch_name}:")
    print(f"    Healthy  — Delta:{h_pow['D']:.3f} Theta:{h_pow['T']:.3f} Alpha:{h_pow['A']:.3f} Beta:{h_pow['B']:.3f} Gamma:{h_pow['G']:.3f}")
    print(f"    Seizure  — Delta:{s_pow['D']:.3f} Theta:{s_pow['T']:.3f} Alpha:{s_pow['A']:.3f} Beta:{s_pow['B']:.3f} Gamma:{s_pow['G']:.3f}")

# Plot band power over time
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

time_axis = np.arange(len(healthy_powers['B'])) * 2  # 2 sec windows

for band, color in zip(['B', 'G', 'A'], ['blue', 'red', 'green']):
    axes[0].plot(time_axis, healthy_powers[band], label=f'{band} band', color=color, alpha=0.7)
axes[0].set_title('Healthy EEG — Band Power Over Time', fontsize=13, color='green')
axes[0].set_ylabel('Relative Power')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

time_axis2 = np.arange(len(seizure_powers['B'])) * 2
for band, color in zip(['B', 'G', 'A'], ['blue', 'red', 'green']):
    axes[1].plot(time_axis2, seizure_powers[band], label=f'{band} band', color=color, alpha=0.7)
axes[1].axvline(x=1467, color='black', linestyle='--', linewidth=2, label='Seizure onset')
axes[1].set_title('Seizure Recording — Band Power Over Time (black line = seizure onset)', fontsize=13, color='red')
axes[1].set_ylabel('Relative Power')
axes[1].set_xlabel('Time (seconds)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/band_powers.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to results/band_powers.png")