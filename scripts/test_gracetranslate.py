import pytest
from gracetranslate import translate


# --- basic translation ---

def test_start_codon():
    assert translate('ATG') == 'M'

def test_simple_peptide():
    assert translate('ATGGGT') == 'MG'

def test_full_orf_with_stop():
    # ATG(M) GGT(G) TAA(*)
    assert translate('ATGGGTTAA') == 'MG'

def test_all_stop_codons_terminate():
    assert translate('ATGTAA') == 'M'
    assert translate('ATGTAG') == 'M'
    assert translate('ATGTGA') == 'M'


# --- length edge cases ---

def test_not_divisible_by_3_trailing_bases_ignored():
    # ATGGGT = MG, then one leftover base 'A' — ignored
    assert translate('ATGGGTA') == 'MG'

def test_not_divisible_by_3_two_trailing_bases_ignored():
    # ATGGGT = MG, then two leftover bases 'AT' — ignored
    assert translate('ATGGGTAT') == 'MG'

def test_empty_sequence():
    assert translate('') == ''

def test_single_base():
    assert translate('A') == ''

def test_two_bases():
    assert translate('AT') == ''


# --- stop codon position ---

def test_stop_codon_at_start():
    assert translate('TAA') == ''

def test_stop_codon_mid_sequence():
    # ATG(M) TAA(*) GGT — stops after M
    assert translate('ATGTAAGGT') == 'M'

def test_no_stop_codon_translates_to_end():
    # ATGGGTCCC = M G P
    assert translate('ATGGGTCCC') == 'MGP'


# --- unknown codons ---

def test_unknown_codon_marked_as_question_mark():
    # ATG(M) NNN(?) GGT(G)
    assert translate('ATGNNNGGTTAA') == 'M?G'

def test_lowercase_input_is_handled():
    assert translate('atgggttaa') == 'MG'


# --- known amino acids spot-check ---

def test_known_codons():
    # Build sequence from individual codons to avoid miscounting
    seq = 'ATG' + 'GGT' + 'CCT' + 'AAC' + 'GAA' + 'TAA'  # M G P N E *
    assert translate(seq) == 'MGPNE'
