#!/usr/bin/env python3
"""CAPITI interrupt: pulse a Pi GPIO pin to abort an in-progress emusynth run.

Wiring (Pi 4):
    Pi GPIO17 (physical pin 11) --[220 ohm]-- Uno D12
                                              |
                                      [10 kohm pulldown to GND]
    Pi GND (physical pin 6)      ------------ Uno GND  (must be common)

Usage:
    python3 capiti_interrupt.py            # single pulse, abort once
    python3 capiti_interrupt.py --hold-ms 100   # adjust pulse width
    python3 capiti_interrupt.py --pin 17        # change GPIO (BCM numbering)

Pulse width >= 50 ms ensures the Uno's polling loop catches the rising edge.
"""
import argparse
import sys
import time

try:
    from gpiozero import DigitalOutputDevice
except ImportError:
    print("ERROR: gpiozero not installed. On Raspberry Pi OS it ships by default;",
          "otherwise: sudo apt install python3-gpiozero", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", type=int, default=17,
                    help="BCM GPIO number (default 17 = physical pin 11)")
    ap.add_argument("--hold-ms", type=int, default=100,
                    help="HIGH pulse width in milliseconds (default 100)")
    args = ap.parse_args()

    # initial=False -> pin starts and stays LOW until we explicitly drive it
    line = DigitalOutputDevice(args.pin, active_high=True, initial_value=False)
    try:
        print(f"CAPITI: pulsing GPIO{args.pin} HIGH for {args.hold_ms} ms")
        line.on()
        time.sleep(args.hold_ms / 1000.0)
        line.off()
        print("CAPITI: pulse done")
    finally:
        line.close()


if __name__ == "__main__":
    main()
