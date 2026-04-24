#!/usr/bin/env python3
import sys
# edge case handling:
# if there are multiple ORFs, it will return all of them, including those that are nested within eachother
# if there is no stop codon, it will return the sequence after the start codon

STOP_CODONS = {'TAA', 'TAG', 'TGA'}

def find_orfs(seq):
    """Return all ORFs in seq. Each ORF starts at ATG and ends at the first
    in-frame stop codon (inclusive), or at the end of the sequence if none."""
    orfs = []
    pos = 0
    while True:
        start = seq.find('ATG', pos)
        if start == -1:
            break
        stop_pos = None
        for i in range(start, len(seq) - 2, 3):
            if seq[i:i+3] in STOP_CODONS:
                stop_pos = i + 3
                break
        if stop_pos is not None:
            orfs.append(seq[start:stop_pos])
        else:
            orfs.append(seq[start:])
        pos = start + 1
    return orfs

def main():
    data = sys.stdin.read().strip()
    if not data:
        return

    results = []
    for seq in data.split(','):
        results.extend(find_orfs(seq.strip().upper()))

    if results:
        sys.stdout.write(','.join(results) + '\n')

if __name__ == '__main__':
    main()
