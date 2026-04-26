"""capiti-watch CLI: live streaming classification during synthesis.

Wiring is identical to `capiti-listen` for input (amidite + TET strobe
GPIO) and `capiti-interrupt` for output (a single GPIO line that pulses
HIGH on trigger).

What it does, in one process:
  1. captures bases as they're synthesized (TET-strobed sampling),
  2. finds the first ATG and translates forward in-frame,
  3. after every N new bases, runs the bundled ONNX model on the AA
     prefix so far,
  4. when the in-set probability stays above --threshold for
     --stability consecutive scorings (and the AA prefix is at least
     --min-k residues long), pulses the interrupt pin and exits.

Default thresholds favor specificity, not earliest-possible-call. The
intent is "we're sure, take action" -- false aborts are worse than a
slightly delayed call. Tune --threshold / --stability / --min-k for
your synthesizer's risk profile.

Usage:
    capiti-watch                       # ab9 model, default thresholds
    capiti-watch --set C               # use the capiti-C bundle
    capiti-watch --no-interrupt -v     # dry run; print verdicts only
    capiti-watch --threshold 0.99 --stability 10
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

from capiti.cli import AVAILABLE_SETS, _bundled, softmax
from capiti.seq import (
    CODON_TABLE, START_CODON, STOP_CODONS, normalize_nt,
)


def _next_codons(nt: str, start: int):
    """Yield (end_offset, aa) for each complete codon starting at
    `start`. Stops at the first stop codon, returning that codon's
    end_offset with aa='*' so the caller can freeze translation."""
    i = start
    while i + 3 <= len(nt):
        codon = nt[i:i + 3]
        if codon in STOP_CODONS:
            yield i + 3, "*"
            return
        yield i + 3, CODON_TABLE.get(codon, "X")
        i += 3


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="capiti-watch",
        description="Live streaming classifier: read synthesizer "
                    "GPIO, score the partial protein, fire the abort "
                    "pin when in-set probability holds above threshold.",
    )
    # GPIO input (mirrors capiti-listen)
    ap.add_argument("--pin-a", type=int, default=5)
    ap.add_argument("--pin-g", type=int, default=6)
    ap.add_argument("--pin-c", type=int, default=13)
    ap.add_argument("--pin-t", type=int, default=19)
    ap.add_argument("--strobe", type=int, default=26)
    ap.add_argument("--done-pin", type=int, default=22,
                    help="set 0 to disable")
    ap.add_argument("--settle-us", type=int, default=200)
    ap.add_argument("--max-bases", type=int, default=0,
                    help="hard cap (default 0 = unbounded)")
    ap.add_argument("--out", help="append each base to this file")
    # Inference
    ap.add_argument("--set", dest="set_name",
                    default=os.environ.get("CAPITI_SET", "ab9"),
                    choices=AVAILABLE_SETS)
    ap.add_argument("--model", default=os.environ.get("CAPITI_MODEL"))
    ap.add_argument("--meta", default=os.environ.get("CAPITI_META"))
    # Streaming verdict
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="fire when p_inset stays above this for "
                         "--stability scorings (default 0.95)")
    ap.add_argument("--min-k", type=int, default=50,
                    help="don't fire below this AA prefix length "
                         "(default 50)")
    ap.add_argument("--stability", type=int, default=5,
                    help="consecutive above-threshold scorings before "
                         "firing (default 5)")
    ap.add_argument("--score-every", type=int, default=3,
                    help="re-score every N new bases (default 3 = once "
                         "per codon)")
    # Action
    ap.add_argument("--interrupt-pin", type=int, default=17)
    ap.add_argument("--interrupt-hold-ms", type=int, default=100)
    ap.add_argument("--no-interrupt", action="store_true",
                    help="dry run: don't pulse the abort pin on trigger")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress per-score line on stderr")
    # Simulation: feed bases from a string / FASTA instead of GPIO.
    # Useful for in-silico demos and pre-hardware testing.
    ap.add_argument("--sim-nt",
                    help="simulate: feed this nucleotide string through "
                         "the same scoring loop, no GPIO. Triggers and "
                         "verdict logic are identical to the live path.")
    ap.add_argument("--sim-fasta",
                    help="simulate: feed each FASTA record's nt sequence "
                         "through the loop in turn (resets state between "
                         "records). `-` reads from stdin.")
    ap.add_argument("--sim-rate-hz", type=float, default=0.0,
                    help="simulated bases per second (default 0 = as fast "
                         "as possible). Set to ~10 to mimic real synth "
                         "pacing for demo videos.")
    args = ap.parse_args(argv)
    sim_mode = bool(args.sim_nt or args.sim_fasta)

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        sys.stderr.write(f"missing dependency: {e}\n")
        return 2
    if not sim_mode:
        try:
            from gpiozero import DigitalInputDevice, DigitalOutputDevice
        except ImportError:
            sys.stderr.write(
                "capiti-watch: gpiozero is not installed.\n"
                "install: pip install 'capiti[pi]'\n"
                "(or run in sim mode with --sim-nt / --sim-fasta)\n"
            )
            return 2

    import json
    model_path = args.model or str(_bundled(args.set_name, "capiti.onnx"))
    meta_path = args.meta or str(_bundled(args.set_name, "capiti.meta.json"))
    if not os.path.exists(model_path):
        sys.stderr.write(f"capiti-watch: model not bundled at {model_path}\n")
        return 2
    with open(meta_path) as fh:
        meta = json.load(fh)
    max_len = meta["max_len"]
    aa_to_idx = meta["vocab"]
    pad_idx = aa_to_idx.get("pad", 0)
    labels = meta["labels"]
    none_idx = meta["none_idx"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("CAPITI_THREADS", "1"))
    so.inter_op_num_threads = 1
    so.log_severity_level = int(os.environ.get("CAPITI_LOG_LEVEL", "3"))
    sess = ort.InferenceSession(model_path, sess_options=so,
                                 providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    if sim_mode:
        base_in = strobe = done_dev = interrupt_line = None
    else:
        base_in = [
            (DigitalInputDevice(args.pin_a, pull_up=None, active_state=True), "A"),
            (DigitalInputDevice(args.pin_g, pull_up=None, active_state=True), "G"),
            (DigitalInputDevice(args.pin_c, pull_up=None, active_state=True), "C"),
            (DigitalInputDevice(args.pin_t, pull_up=None, active_state=True), "T"),
        ]
        strobe = DigitalInputDevice(args.strobe, pull_up=None, active_state=True)
        done_dev = (DigitalInputDevice(args.done_pin, pull_up=None,
                                        active_state=True)
                    if args.done_pin else None)
        interrupt_line = DigitalOutputDevice(
            args.interrupt_pin, active_high=True, initial_value=False)

    nt_chars: list[str] = []
    aa_chars: list[str] = []
    state = {
        "atg_offset": None,
        "next_codon_start": None,
        "translation_frozen": False,
        "last_score_at": 0,
        "consec_above": 0,
        "triggered": False,
    }
    out_fh = open(args.out, "a", buffering=1) if args.out else None
    done = threading.Event()
    settle = max(args.settle_us, 0) / 1_000_000.0

    x_idx = aa_to_idx.get("X", pad_idx)
    base_buf = [pad_idx] * max_len  # reusable encode buffer

    def encode_into(seq):
        # In-place encode + pad. Avoids reallocating max_len ints per score.
        for i in range(max_len):
            base_buf[i] = pad_idx
        for i, c in enumerate(seq[:max_len]):
            base_buf[i] = aa_to_idx.get(c, x_idx)
        return base_buf

    def maybe_score(force=False):
        if state["triggered"]:
            return
        n = len(nt_chars)
        if not force and n - state["last_score_at"] < args.score_every:
            return
        state["last_score_at"] = n

        # Track ATG and translate forward in-frame.
        if state["atg_offset"] is None:
            joined = "".join(nt_chars)
            atg = joined.find(START_CODON)
            if atg < 0:
                return
            state["atg_offset"] = atg
            state["next_codon_start"] = atg

        if not state["translation_frozen"]:
            for end, aa in _next_codons("".join(nt_chars),
                                          state["next_codon_start"]):
                state["next_codon_start"] = end
                if aa == "*":
                    state["translation_frozen"] = True
                    break
                aa_chars.append(aa)

        K = len(aa_chars)
        if K < args.min_k:
            return

        x = np.asarray([encode_into(aa_chars)], dtype=np.int64)
        logits = sess.run(None, {inp_name: x})[0]
        probs = softmax(logits)[0]
        p_inset = float(1.0 - probs[none_idx])
        masked = probs.copy(); masked[none_idx] = -1.0
        top_idx = int(masked.argmax())
        top_label = labels[top_idx]

        above = p_inset >= args.threshold
        state["consec_above"] = state["consec_above"] + 1 if above else 0
        will_fire = state["consec_above"] >= args.stability

        if not args.quiet:
            tag = ("FIRE" if will_fire
                   else ("HOT" if above else "ok"))
            sys.stderr.write(
                f"[K={K:3d} bases={n:4d} top={top_label} "
                f"p={p_inset:.3f} run={state['consec_above']} {tag}]\n"
            )
            sys.stderr.flush()

        if will_fire:
            state["triggered"] = True
            will_pulse = (interrupt_line is not None) and (not args.no_interrupt)
            sys.stderr.write(
                f"capiti-watch: TRIGGER at K={K} (top={top_label}, "
                f"p_inset={p_inset:.3f}); "
                f"{'firing interrupt' if will_pulse else 'no pulse (sim or dry run)'}\n"
            )
            if will_pulse:
                interrupt_line.on()
                time.sleep(args.interrupt_hold_ms / 1000.0)
                interrupt_line.off()
            done.set()

    def process_base(base):
        nt_chars.append(base)
        if out_fh is not None:
            out_fh.write(base)
        maybe_score()
        if args.max_bases and len(nt_chars) >= args.max_bases:
            done.set()

    def on_strobe():
        if settle:
            time.sleep(settle)
        high = [letter for dev, letter in base_in if dev.value]
        if len(high) == 1:
            base = high[0]
        else:
            base = "N"
            sys.stderr.write(
                f"capiti-watch[{len(nt_chars)+1}]: "
                f"strobe with {'no' if not high else 'multiple'} amidite "
                f"line(s) HIGH ({'+'.join(high) or '-'}) -> N\n"
            )
        process_base(base)

    signal.signal(signal.SIGINT, lambda *_: done.set())
    if not sim_mode:
        strobe.when_activated = on_strobe
        if done_dev is not None:
            done_dev.when_activated = lambda: done.set()

    src_label = ("sim" if sim_mode
                 else f"GPIO strobe={args.strobe}")
    sys.stderr.write(
        f"capiti-watch: set={args.set_name} src={src_label} "
        f"threshold={args.threshold} min_k={args.min_k} "
        f"stability={args.stability} score_every={args.score_every}b "
        f"interrupt="
        f"{'OFF' if (sim_mode or args.no_interrupt) else f'GPIO{args.interrupt_pin}'}"
        f"\n"
    )

    def feed_sim_records():
        """Yield (record_name, nt_string) tuples for every sim source."""
        if args.sim_nt:
            yield ("sim", normalize_nt(args.sim_nt))
        if args.sim_fasta:
            from capiti.seq import _iter_fasta
            if args.sim_fasta == "-":
                for n, s in _iter_fasta(sys.stdin):
                    yield (n, normalize_nt(s))
            else:
                with open(args.sim_fasta) as fh:
                    for n, s in _iter_fasta(fh):
                        yield (n, normalize_nt(s))

    def reset_state():
        nt_chars.clear()
        aa_chars.clear()
        state.update(atg_offset=None, next_codon_start=None,
                     translation_frozen=False, last_score_at=0,
                     consec_above=0, triggered=False)

    try:
        if sim_mode:
            tick = 1.0 / args.sim_rate_hz if args.sim_rate_hz > 0 else 0.0
            for name, nt in feed_sim_records():
                if not args.quiet:
                    sys.stderr.write(
                        f"capiti-watch[sim]: feeding {name} "
                        f"({len(nt)} bases)\n")
                reset_state()
                done.clear()
                for ch in nt:
                    if done.is_set():
                        break
                    process_base(ch)
                    if tick:
                        time.sleep(tick)
                # End-of-record: flush the final partial codon score
                # so the user sees the ending verdict even if no
                # rescoring boundary fell exactly at the last base.
                maybe_score(force=True)
                sys.stderr.write(
                    f"capiti-watch[sim]: {name} -> "
                    f"K={len(aa_chars)} "
                    f"{'TRIGGERED' if state['triggered'] else 'no trigger'}\n"
                )
        else:
            done.wait()
            time.sleep(max(settle, 0.05))
    finally:
        if not sim_mode:
            strobe.close()
            if done_dev is not None:
                done_dev.close()
            for dev, _ in base_in:
                dev.close()
            interrupt_line.close()
        if out_fh is not None:
            out_fh.close()

    sys.stderr.write(
        f"capiti-watch: stopped after {len(nt_chars)} bases, "
        f"K={len(aa_chars)}, "
        f"{'TRIGGERED' if state['triggered'] else 'no trigger'}\n"
    )
    return 0 if state["triggered"] else 1


if __name__ == "__main__":
    sys.exit(main())
