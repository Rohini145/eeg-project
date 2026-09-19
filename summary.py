with open('data/chbmit/chb01-summary.txt', 'r') as f:
    content = f.read()

# Print info for all three files
for filename in ['chb01_01', 'chb01_02', 'chb01_03']:
    print(f"\n{'='*40}")
    print(f"File: {filename}")
    
    # Find section for this file
    start = content.find(filename)
    if start != -1:
        section = content[start:start+300]
        print(section)
    else:
        print("Not found in summary")