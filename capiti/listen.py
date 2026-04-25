"""capiti-listen CLI: read amidite/activator pulses from a Pi GPIO and
reconstruct the nucleotide sequence the synthesizer is delivering.

Wiring (Pi 4, BCM numbering):
    Uno D2  -> GPIO5  (29)   A (amidite)
    Uno D3  -> GPIO6  (31)   G (amidite)
    Uno D4  -> GPIO13 (33)   C (amidite)
    Uno D5  -> GPIO19 (35)   T (amidite)
    Uno D6  -> GPIO26 (37)   TET (activator / strobe)
    Uno D7  -> GPIO22 (15)   DONE (rising edge stops capture; optional)
    Uno GND -> Pi GND        common ground required.

Each base line is held HIGH while its amidite valve is open; the TET line
pulses HIGH once per coupling cycle. We treat TET as the clock: on every
TET rising edge, we sample the four base lines, and whichever is HIGH at
that instant is the base being incorporated. Sampling at the strobe rather
than counting edges on the base lines means homopolymers are read
correctly (back-to-back identical bases just hold the same line HIGH
across multiple TET pulses).

Usage:
    capiti-listen                      # stream to stdout, DONE or Ctrl-C stops
    capiti-listen --out seq.txt        # also append each base to a file
    capiti-listen --max-len 200        # additionally stop after 200 bases
    capiti-listen --done-pin 0         # disable DONE; rely on Ctrl-C / max-len
    capiti-listen --quiet              # only print final sequence at end

Pipe into the classifier:
    capiti-listen --quiet | capiti --stdin
"""
from __future__ import annotations
import argparse
import signal
import sys
import threading
import time


# (BCM GPIO, base letter)
DEFAULT_BASE_PINS = [
    (5,  "A"),
    (6,  "G"),
    (13, "C"),
    (19, "T"),
]
DEFAULT_STROBE_PIN = 26


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="capiti-listen",
        description="Reconstruct a DNA sequence from amidite + activator "
                    "pulses on Raspberry Pi GPIO inputs.",
    )
    ap.add_argument("--pin-a", type=int, default=5,  help="BCM pin for A (default 5)")
    ap.add_argument("--pin-g", type=int, default=6,  help="BCM pin for G (default 6)")
    ap.add_argument("--pin-c", type=int, default=13, help="BCM pin for C (default 13)")
    ap.add_argument("--pin-t", type=int, default=19, help="BCM pin for T (default 19)")
    ap.add_argument("--strobe", type=int, default=26,
                    help="BCM pin for TET activator strobe (default 26)")
    ap.add_argument("--done-pin", type=int, default=22,
                    help="BCM pin for DONE signal; rising edge stops capture "
                         "(default 22; set 0 to disable)")
    ap.add_argument("--out", help="append each base to this file as it arrives")
    ap.add_argument("--max-len", type=int, default=0,
                    help="stop after N bases (default 0 = unbounded)")
    ap.add_argument("--settle-us", type=int, default=200,
                    help="microseconds to wait after TET edge before "
                         "sampling base pins (default 200)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-base output; only print final sequence")
    args = ap.parse_args(argv)

    try:
        from gpiozero import DigitalInputDevice
    except ImportError:
        sys.stderr.write(
            "capiti-listen: gpiozero is not installed.\n"
            "install: pip install 'capiti[pi]'\n"
        )
        return 2

    base_pins = [
        (args.pin_a, "A"),
        (args.pin_g, "G"),
        (args.pin_c, "C"),
        (args.pin_t, "T"),
    ]

    # External pulldowns are assumed on every line (matching the interrupt
    # wiring). pull_up=None, active_state=True leaves the internal resistor disabled.
    base_in = [(DigitalInputDevice(pin, pull_up=None, active_state=True), letter)
               for pin, letter in base_pins]
    strobe = DigitalInputDevice(args.strobe, pull_up=None, active_state=True)
    done_dev = (DigitalInputDevice(args.done_pin, pull_up=None, active_state=True)
                if args.done_pin else None)

    seq = []
    out_fh = open(args.out, "a", buffering=1) if args.out else None
    done = threading.Event()
    settle = max(args.settle_us, 0) / 1_000_000.0

    def on_strobe():
        if settle:
            time.sleep(settle)
        high = [letter for dev, letter in base_in if dev.value]
        if len(high) == 1:
            base = high[0]
        elif not high:
            base = "N"
            sys.stderr.write(
                f"\ncapiti-listen[{len(seq)+1}]: TET pulse with no amidite "
                f"line HIGH -> N (check wiring / settle-us)\n"
            )
        else:
            base = "N"
            sys.stderr.write(
                f"\ncapiti-listen[{len(seq)+1}]: TET pulse with multiple "
                f"amidite lines HIGH ({'+'.join(high)}) -> N\n"
            )
        seq.append(base)
        if out_fh is not None:
            out_fh.write(base)
        if not args.quiet:
            sys.stdout.write(base)
            sys.stdout.flush()
        if args.max_len and len(seq) >= args.max_len:
            done.set()

    strobe.when_activated = on_strobe
    if done_dev is not None:
        done_dev.when_activated = lambda: done.set()

    if not args.quiet:
        done_msg = f" DONE=GPIO{args.done_pin}" if done_dev is not None else ""
        sys.stderr.write(
            f"capiti-listen: watching GPIO{args.strobe} (TET); "
            f"bases on A=GPIO{args.pin_a} G=GPIO{args.pin_g} "
            f"C=GPIO{args.pin_c} T=GPIO{args.pin_t}.{done_msg} "
            f"Ctrl-C to stop.\n"
        )

    signal.signal(signal.SIGINT, lambda *_: done.set())
    try:
        done.wait()
        # Drain: if DONE/SIGINT raced an in-flight strobe callback, give it
        # a chance to finish appending before we tear the device down.
        time.sleep(max(settle, 0.05))
    finally:
        strobe.close()
        if done_dev is not None:
            done_dev.close()
        for dev, _ in base_in:
            dev.close()
        if out_fh is not None:
            out_fh.close()

    if not args.quiet:
        sys.stdout.write("\n")
    sys.stderr.write(f"capiti-listen: captured {len(seq)} bases\n")
    if args.quiet:
        print("".join(seq))

    return 0


if __name__ == "__main__":
    sys.exit(main())
