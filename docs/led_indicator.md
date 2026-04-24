# LED indicator for capiti (Raspberry Pi)

Small companion CLI to chain off capiti's exit code: TRUE (in-set)
lights one LED, FALSE (not in-set) lights another, for 5s.

## Wiring

Two LEDs, each with a current-limiting resistor. Pick any two BCM GPIO
pins; example uses GPIO17 (header pin 11) and GPIO27 (header pin 13).

```
GPIO17 (pin 11) -- 330 ohm -- (+)GREEN LED(-) -- GND (pin 6)
GPIO27 (pin 13) -- 330 ohm -- (+)RED   LED(-) -- GND (pin 6)
```

- 220-330 ohm for a standard 5mm LED on the Pi's 3.3V logic (Pi IO is
  3.3V only; do not pull from 5V).
- Long leg of the LED = anode = toward the GPIO-side of the resistor.
- Any ground pin works (6, 9, 14, 20, 25, 30, 34, 39 are GND).

## Invocation pattern (preferred)

Keep capiti and the LED tool orthogonal; chain on exit code:

```
capiti ATGCGT... ; capiti-led $?
```

Arg is capiti's exit code: `0` -> TRUE LED, anything else -> FALSE LED.
Avoid `capiti ... && on || off`: `||` fires on any failure, including
bad args or a crash, which is the wrong semantic for the FALSE case.

Optional flags to consider:

- `--hold <sec>` (default 5)
- `--bg` (return immediately; LEDs fade in the background via
  Popen/setsid)
- `--true-pin` / `--false-pin` (default 17 / 27, BCM)
- a brief self-test at startup (light both LEDs for ~200ms) to make
  unresponsive wiring obvious on a headless deployment

## Software

Use `gpiozero`. Standard, works on Pi 3/4/5, automatically uses
`lgpio`/`libgpiod` on Pi 5 + Bookworm and `RPi.GPIO` on older. User
must be in the `gpio` group (no root needed); otherwise `sudo`.

Implementation sketch:

```python
# capiti/led.py
import argparse, sys, time
from gpiozero import LED

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exit_code", type=int)
    ap.add_argument("--true-pin", type=int, default=17)
    ap.add_argument("--false-pin", type=int, default=27)
    ap.add_argument("--hold", type=float, default=5.0)
    args = ap.parse_args()
    pin = args.true_pin if args.exit_code == 0 else args.false_pin
    led = LED(pin)
    led.on()
    time.sleep(args.hold)
    led.off()

if __name__ == "__main__":
    sys.exit(main())
```

## Packaging

Only makes sense on a Pi, so gate with an extras group. In
`pyproject.toml`:

```toml
[project.optional-dependencies]
pi = ["gpiozero>=2"]

[project.scripts]
capiti = "capiti.cli:main"
capiti-led = "capiti.led:main"
```

Install on the Pi with `pip install 'capiti[pi]'`. Non-Pi environments
don't pull `gpiozero`.

## Tradeoffs

- `gpiozero` needs the `gpio` group membership (or `sudo`). Add the
  user with `sudo usermod -aG gpio $USER` and log out/in once.
- On Pi 5 with Bookworm, `RPi.GPIO` is deprecated; `gpiozero` papers
  over that, but mention it in the Pi install docs so nobody pins
  `RPi.GPIO` directly.
- 5s is a blocking sleep in the simple version. Fine interactively,
  annoying in scripts - add `--bg` (spawn a detached child and return
  0 immediately) if the user wants to keep piping.
- If either pin is already in use by another process, `gpiozero` will
  raise `GPIOPinInUse`. Catch and give a clear error.

## Testing without hardware

`gpiozero` has a `MockFactory` backend that lets unit tests run without
a Pi. `from gpiozero.pins.mock import MockFactory; Device.pin_factory
= MockFactory()` in a test conftest. Good for CI coverage of the
argparse / exit-code logic.
