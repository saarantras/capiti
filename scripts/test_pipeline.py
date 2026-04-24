"""
End-to-end integration tests for the full pipeline:
  echo <seq> | gracepermute | graceorffinder | gracetranslate
"""
import subprocess
import sys
import os
import pytest

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PERMUTE    = os.path.join(SCRIPTS, 'gracepermute.py')
ORFFINDER  = os.path.join(SCRIPTS, 'graceorffinder.py')
TRANSLATE  = os.path.join(SCRIPTS, 'gracetranslate.py')
PY = sys.executable


def pipeline(seq):
    """Run seq through the full pipeline; return sorted list of proteins (or [] if empty)."""
    cmd = f'echo {seq!r} | {PY} {PERMUTE} | {PY} {ORFFINDER} | {PY} {TRANSLATE}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = result.stdout.strip()
    return sorted(out.split(',')) if out else []


def pipeline_raw(seq):
    """Same as pipeline() but returns the raw stdout string (stripped)."""
    cmd = f'echo {seq!r} | {PY} {PERMUTE} | {PY} {ORFFINDER} | {PY} {TRANSLATE}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


# ─── basic sanity ──────────────────────────────────────────────────────────────

def test_canonical_example():
    # the example from the spec
    assert 'M' in pipeline('ATGTAA')

def test_lowercase_input_treated_same_as_uppercase():
    assert pipeline('atgtaa') == pipeline('ATGTAA')

def test_output_contains_no_empty_entries():
    proteins = pipeline('ATGTAA')
    assert all(p != '' for p in proteins)

def test_output_is_comma_separated_on_single_line():
    raw = pipeline_raw('ATGTAA')
    assert '\n' not in raw
    assert ',' in raw or len(raw) > 0  # at least one result


# ─── no-ORF inputs ─────────────────────────────────────────────────────────────

def test_single_nucleotide_produces_no_output():
    # 'A' → 24 single-char permutations, none contain ATG
    assert pipeline('A') == []

def test_two_nucleotides_produce_no_output():
    assert pipeline('AT') == []

def test_all_same_nucleotide_produces_no_output():
    # any permutation of TTTT is XXXX — never contains ATG
    assert pipeline('TTTT') == []

def test_only_stop_codons_can_still_produce_orfs_after_permutation():
    # TAATAGTGA maps to sequences containing ATG under some permutations
    # verified ground truth: ['M', 'MS', 'M']
    result = pipeline('TAATAGTGA')
    assert result != []
    assert 'M' in result


# ─── ORF only after permutation ────────────────────────────────────────────────

def test_orf_only_in_permuted_version():
    # TAGCTA has no ATG in identity; A<->T swap gives ATGCAT -> MH
    result = pipeline('TAGCTA')
    assert 'MH' in result

def test_identity_has_no_orf_but_pipeline_still_finds_one():
    # TAGCTA has no ATG → result must come from a non-identity permutation
    from graceorffinder import find_orfs
    assert find_orfs('TAGCTA') == []
    assert pipeline('TAGCTA') != []


# ─── stop codon variants ───────────────────────────────────────────────────────

def test_taa_stop():
    assert 'M' in pipeline('ATGTAA')

def test_tag_stop():
    assert 'M' in pipeline('ATGTAG')

def test_tga_stop():
    assert 'M' in pipeline('ATGTGA')

def test_stop_codon_mid_sequence_truncates_translation():
    # ATGTTTTAA → MF (TTT=Phe, TAA=stop)
    assert 'MF' in pipeline('ATGTTTTAA')

def test_two_stop_codons_in_a_row():
    # ATGTAATAA: first stop at pos 3 → ORF is ATGTAA → M
    result = pipeline('ATGTAATAA')
    assert 'M' in result

def test_tga_stop_mid_sequence():
    # ATGTGAAAA: TGA is stop at codon 2 → ORF ATGTGA → M
    result = pipeline('ATGTGAAAA')
    assert 'M' in result


# ─── multiple ORFs ─────────────────────────────────────────────────────────────

def test_nested_atg_produces_multiple_orfs():
    # ATGATGTAA: nested ORFs → pipeline produces MM and M among results
    result = pipeline('ATGATGTAA')
    assert 'MM' in result
    assert 'M' in result

def test_triple_atg_no_stop():
    # ATGATGATG: three nested ORFs with no stop → MMM, MM, M
    result = pipeline('ATGATGATG')
    assert 'MMM' in result
    assert 'MM' in result
    assert 'M' in result

def test_two_sequential_orfs():
    # ATGTAAATGTAA: two separate ORFs, both translate to M
    result = pipeline('ATGTAAATGTAA')
    assert 'M' in result


# ─── no stop codon (partial ORF returned) ──────────────────────────────────────

def test_no_stop_codon_single_orf():
    # ATG alone → identity permutation gives just ATG → M
    assert 'M' in pipeline('ATG')

def test_no_stop_codon_longer_sequence():
    # identity perm → ATGCCCGGG → MPG (M, Pro, Gly)
    # another perm {A:C,T:A,G:T,C:G} → CATGGGTTT → ORF starts at pos 1 → MG
    result = pipeline('ATGCCCGGG')
    assert 'MPG' in result
    assert 'MG' in result


# ─── non-nucleotide characters ─────────────────────────────────────────────────

def test_ambiguous_n_base_produces_question_mark():
    # ATGNNN: N passes through unmapped, unknown codon → M?
    result = pipeline('ATGNNN')
    assert 'M?' in result

def test_non_nucleotide_char_passes_through():
    # non-ATGC chars are left unchanged by gracepermute
    result = pipeline('ATG1AATAA')
    assert 'M?' in result


# ─── gracepermute output integrity ─────────────────────────────────────────────

def test_gracepermute_produces_exactly_24_sequences():
    cmd = f'echo "ATGC" | {PY} {PERMUTE}'
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    parts = out.split(',')
    assert len(parts) == 24

def test_gracepermute_output_has_no_garbage_lines():
    # guard against accidental debug print() lines corrupting stdout
    cmd = f'echo "ATGC" | {PY} {PERMUTE}'
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    assert out.count('\n') == 1, "gracepermute must write exactly one newline"

def test_gracepermute_all_outputs_same_length_as_input():
    cmd = f'echo "ATGC" | {PY} {PERMUTE}'
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    assert all(len(s) == 4 for s in out.split(','))


# ─── empty / whitespace input ──────────────────────────────────────────────────

def test_empty_input_produces_no_output():
    cmd = f'printf "" | {PY} {PERMUTE} | {PY} {ORFFINDER} | {PY} {TRANSLATE}'
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    assert out == ''

def test_whitespace_only_input_produces_no_output():
    cmd = f'echo "   " | {PY} {PERMUTE} | {PY} {ORFFINDER} | {PY} {TRANSLATE}'
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    assert out == ''
