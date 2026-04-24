#!/usr/bin/env python3
import sys
import itertools

NUCLEOTIDES = ['A', 'T', 'G', 'C']

def apply_mapping(seq, mapping):
    return ''.join(mapping.get(c, c) for c in seq)

def main():
    seq = sys.stdin.read().strip().upper()
    if not seq:
        return
    first = True
    for perm in itertools.permutations(NUCLEOTIDES):
        #print(perm)
        mapping = dict(zip(NUCLEOTIDES, perm))
        result = apply_mapping(seq, mapping)
        if not first:
            sys.stdout.write(',')
        sys.stdout.write(result)
        sys.stdout.flush()
        first = False
    sys.stdout.write('\n')

if __name__ == '__main__':
    main()
