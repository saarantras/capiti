import pytest
from graceorffinder import find_orfs


# --- basic cases ---

def test_simple_orf_with_stop():
    assert find_orfs('ATGTAA') == ['ATGTAA']

def test_simple_orf_with_tag():
    assert find_orfs('ATGTAG') == ['ATGTAG']

def test_simple_orf_with_tga():
    assert find_orfs('ATGTGA') == ['ATGTGA']

def test_no_atg():
    assert find_orfs('TTTCCCGGG') == []

def test_empty_sequence():
    assert find_orfs('') == []


# --- no stop codon ---

def test_no_stop_codon_returns_rest_of_sequence():
    assert find_orfs('ATGCCC') == ['ATGCCC']

def test_no_stop_codon_odd_length_returns_rest():
    # length not a multiple of 3 after ATG — still return from ATG to end
    assert find_orfs('ATGCC') == ['ATGCC']

def test_atg_at_end_no_codons():
    # ATG with nothing after it
    assert find_orfs('XXXATG') == ['ATG']


# --- multiple ORFs ---

def test_two_sequential_orfs():
    # ATG@0 → ATGTAA; ATG@6 → ATGCCC (no stop, returns rest)
    assert find_orfs('ATGTAAATGCCC') == ['ATGTAA', 'ATGCCC']

def test_two_non_overlapping_orfs_with_gap():
    # ATG@0 → ATGTAA; ATG@9 → ATGTAG
    assert find_orfs('ATGTAACCCATGTAG') == ['ATGTAA', 'ATGTAG']

def test_nested_orf():
    # ATG at 0 in-frame: ATG ATG TAA → ATGATGTAA
    # ATG at 3 in-frame: ATG TAA → ATGTAA
    assert find_orfs('ATGATGTAA') == ['ATGATGTAA', 'ATGTAA']


# --- stop codon framing ---

def test_stop_codon_out_of_frame_is_ignored():
    # ATGXTAA: stop is out of frame from ATG, no in-frame stop → return rest
    assert find_orfs('ATGXTAA') == ['ATGXTAA']

def test_stop_codon_in_frame_only():
    # ATGNNNTAA: TAA is in-frame (positions 0,3,6)
    assert find_orfs('ATGNNNTAA') == ['ATGNNNTAA']


# --- ATG inside stop codon region ---

def test_atg_after_stop_is_found():
    result = find_orfs('ATGTAAATG')
    assert 'ATGTAA' in result
    assert 'ATG' in result  # last ATG has no stop, returns rest


# --- non-nucleotide passthrough ---

def test_non_nucleotide_chars_passthrough():
    # unknown chars are left alone; stop codon still detected
    assert find_orfs('XATGXTXTAA') == ['ATGXTXTAA']
