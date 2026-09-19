import mne
import numpy as np
import matplotlib.pyplot as plt

# Load the file that HAS the seizure
raw = mne.io.read_raw_edf('data/chbmit/chb01_02.edf', preload=True, verbose=False)

sfreq = raw.info['sfreq']  # 256 Hz

# Seizure is at 1467 seconds
seizure_start = 1467
seizure_end   = 1497  # 30 seconds of seizure

# Healthy segment — 30 seconds well before seizure
healthy_start = 1000
healthy_end   = 1030

# Extract segments
data = raw.get_data()  # shape: (23, samples)

healthy_seg = data[:, int(healthy_start*sfreq) : int(healthy_end*sfreq)]
seizure_seg = data[:, int(seizure_start*sfreq) : int(seizure_end*sfreq)]

print(f"Healthy segment shape: {healthy_seg.shape}")
print(f"Seizure segment shape: {seizure_seg.shape}")

# Plot both side by side
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot channel FZ-CZ (index 16) — good central channel
ch = 16

axes[0].plot(np.linspace(0, 30, healthy_seg.shape[1]), 
             healthy_seg[ch] * 1e6)  # convert to microvolts
axes[0].set_title('Healthy EEG — 30 seconds (no seizure)', fontsize=14, color='green')
axes[0].set_ylabel('Amplitude (µV)')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylim(-200, 200)
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.linspace(0, 30, seizure_seg.shape[1]), 
             seizure_seg[ch] * 1e6, color='red')
axes[1].set_title('Seizure EEG — 30 seconds (ictal activity)', fontsize=14, color='red')
axes[1].set_ylabel('Amplitude (µV)')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylim(-200, 200)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Healthy vs Seizure EEG — Patient chb01', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/healthy_vs_seizure.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to results/healthy_vs_seizure.png")
print(f"\nHealthy segment — mean amplitude: {np.mean(np.abs(healthy_seg[ch]*1e6)):.2f} µV")
print(f"Seizure segment — mean amplitude: {np.mean(np.abs(seizure_seg[ch]*1e6)):.2f} µV")