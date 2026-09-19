import os
import urllib.request
import time

# All 23 CHB-MIT patients
patients = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05',
    'chb06', 'chb07', 'chb08', 'chb09', 'chb10',
    'chb11', 'chb12', 'chb13', 'chb14', 'chb15',
    'chb16', 'chb17', 'chb18', 'chb19', 'chb20',
    'chb21', 'chb22', 'chb23'
]

base_url = "https://physionet.org/files/chbmit/1.0.0"
base_dir = "data/chbmit"
os.makedirs(base_dir, exist_ok=True)

# First just download the summary files for all patients
# These are tiny (a few KB each) and tell us exactly
# which files contain seizures — so we download smartly
print("=== Step 1: Download summary files for all patients ===")
print("This tells us which recordings have seizures\n")

success = []
failed  = []

for patient in patients:
    out_dir = f"{base_dir}/{patient}"
    os.makedirs(out_dir, exist_ok=True)
    
    summary_url  = f"{base_url}/{patient}/{patient}-summary.txt"
    summary_path = f"{out_dir}/{patient}-summary.txt"
    
    if os.path.exists(summary_path):
        print(f"  {patient} — summary already exists, skipping")
        success.append(patient)
        continue
    
    try:
        urllib.request.urlretrieve(summary_url, summary_path)
        print(f"  {patient} — ✅ summary downloaded")
        success.append(patient)
        time.sleep(0.5)  # be polite to PhysioNet server
    except Exception as e:
        print(f"  {patient} — ❌ failed: {e}")
        failed.append(patient)

print(f"\n✅ Success: {len(success)} patients")
print(f"❌ Failed:  {len(failed)} patients")
if failed:
    print(f"   Failed list: {failed}")

# Now parse all summaries to find which files have seizures
print("\n=== Step 2: Parse summaries — find seizure files ===\n")

seizure_map = {}  # patient -> list of (filename, start_sec, end_sec)

for patient in success:
    summary_path = f"{base_dir}/{patient}/{patient}-summary.txt"
    if not os.path.exists(summary_path):
        continue
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    seizure_map[patient] = []
    lines = content.split('\n')
    current_file = None
    
    for i, line in enumerate(lines):
        if 'File Name:' in line:
            current_file = line.split(':')[-1].strip().replace('.edf', '')
        if 'Seizure Start Time:' in line and current_file:
            try:
                start = int(''.join(filter(str.isdigit, line.split(':')[-1])))
                # Look for end time in next few lines
                for j in range(i+1, min(i+4, len(lines))):
                    if 'Seizure End Time:' in lines[j]:
                        end = int(''.join(filter(str.isdigit, lines[j].split(':')[-1])))
                        seizure_map[patient].append((current_file, start, end))
                        break
            except:
                pass

# Print seizure summary
total_seizures = 0
print(f"{'Patient':<10} {'Seizures':<10} {'Files with seizures'}")
print("-" * 60)
for patient in sorted(seizure_map.keys()):
    seizures = seizure_map[patient]
    files = list(set([s[0] for s in seizures]))
    print(f"{patient:<10} {len(seizures):<10} {', '.join(files[:3])}{'...' if len(files)>3 else ''}")
    total_seizures += len(seizures)

print("-" * 60)
print(f"{'TOTAL':<10} {total_seizures:<10} seizures across {len(seizure_map)} patients")

# Save seizure map for next script
import json
with open('data/seizure_map.json', 'w') as f:
    json.dump(seizure_map, f, indent=2)
print("\nSeizure map saved to data/seizure_map.json")
print("Run this again after logging into PhysioNet if any patients failed.")