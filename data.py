import mne
import matplotlib.pyplot as plt

# Load the EEG file
raw = mne.io.read_raw_edf('data/chbmit/chb01_02.edf', preload=True, verbose=False)

# Print basic info
print("=== EEG Recording Info ===")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds ({raw.times[-1]/60:.1f} minutes)")
print(f"Total data shape: {raw.get_data().shape}  (channels x samples)")

# Plot first 10 seconds
print("\nOpening EEG plot — close the window to continue...")
raw.plot(duration=10, n_channels=10, scalings='auto', title='chb01_01 — First 10 seconds')
plt.show()