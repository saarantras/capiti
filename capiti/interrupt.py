"""capiti-interrupt CLI: pulse a Pi GPIO pin to abort an in-progress emusynth run.

Wiring (Pi 4):
    Pi GPIO17 (physical pin 11) --[220 ohm]-- Uno D12
                                              |
                                      [10 kohm pulldown to GND]
    Pi GND (physical pin 6)      ------------ Uno GND  (must be common)

Usage:
    capiti-interrupt                  # single 100 ms HIGH pulse on GPIO17
    capiti-interrupt --hold-ms 50     # adjust pulse width
    capiti-interrupt --pin 17         # change GPIO (BCM numbering)

Pulse width >= 50 ms ensures the Uno's polling loop catches the rising edge.
"""
from __future__ import annotations
import argparse
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="capiti-interrupt",
        description="Pulse a Raspberry Pi GPIO pin HIGH to abort an "
                    "in-progress emusynth run.",
    )
    ap.add_argument("--pin", type=int, default=17,
                    help="BCM GPIO number (default 17 = physical pin 11)")
    ap.add_argument("--hold-ms", type=int, default=100,
                    help="HIGH pulse width in milliseconds (default 100)")
    args = ap.parse_args(argv)

    try:
        from gpiozero import DigitalOutputDevice
    except ImportError:
        sys.stderr.write(
            "capiti-interrupt: gpiozero is not installed.\n"
            "install: pip install 'capiti[pi]'\n"
            "(also requires liblgpio-dev on the host: "
            "sudo apt install -y liblgpio-dev)\n"
        )
        return 2

    line = DigitalOutputDevice(args.pin, active_high=True, initial_value=False)
    try:
        print(f"capiti-interrupt: pulsing GPIO{args.pin} HIGH for {args.hold_ms} ms")
        line.on()
        time.sleep(args.hold_ms / 1000.0)
        line.off()
        print("capiti-interrupt: pulse done")
    finally:
        line.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
