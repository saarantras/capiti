# emusynth -- Pi side (CAPITI)

CAPITI is the Raspberry Pi software that talks to the emusynth Arduino Uno.
For now, the only piece here is the **abort line** -- a one-shot script that
pulses Uno D12 HIGH to kill an in-progress synthesis.

The decoder (read GPIO, reconstruct the nucleotide sequence) lives in this
directory next, but isn't written yet.

## Files

| File                  | Purpose                                                     |
|-----------------------|-------------------------------------------------------------|
| `capiti_interrupt.py` | Pulse GPIO17 HIGH for ~100ms; aborts the Uno's current run. |

## One-time setup on the Pi

Raspberry Pi OS (Bookworm or newer) ships with `gpiozero` and the Pi 4's
GPIO chip already wired through `lgpio`. If you're on a clean install:

```bash
sudo apt update
sudo apt install -y python3-gpiozero python3-lgpio
```

That's it. No virtualenv needed for this one script.

## Wiring (must already be done on the Uno side)

```
Pi GPIO17 (physical pin 11) --[220 ohm]-- Uno D12
                                          |
                                  [10 kohm pulldown to Uno GND]
Pi GND (physical pin 6)      ------------ Uno GND  (common ground)
```

10k pulldown is mandatory -- without it, an unconnected or rebooting Pi
floats the input and the Uno will randomly abort.

## Test it

1. On the Mac, in Arduino IDE Serial Monitor, start a synthesis:
   ```
   LOAD ACGTACGTACGT
   DIL 0.05
   RUN
   ```
   Watch LCD/LEDs cycling.

2. On the Pi, while the synthesis is running:
   ```bash
   python3 capiti_interrupt.py
   ```

3. Within ~50 ms, the Uno LCD should switch to:
   ```
   CAPITI
   INTERRUPT
   ```
   Serial Monitor on the Mac should print `CAPITI INTERRUPT`. All valve LEDs
   go dark. Synthesis is aborted -- no further `NEXT` replies for streams.

## Troubleshooting

- **Script errors with "module gpiozero not found"**: install with the apt
  command above.
- **Script runs but Uno doesn't react**: check (a) common ground between Pi
  and Uno, (b) the 10k pulldown is actually bridging D12 to GND, (c) the 220
  ohm series resistor is in line (not floating).
- **Uno aborts immediately on `RUN`, before the Pi script is invoked**:
  pulldown isn't connecting properly -- D12 is floating HIGH. Recheck which
  breadboard rows the resistor legs are in.
- **Uno aborts at random times**: noise pickup, usually a missing or
  intermittent ground. Verify the Pi-to-Uno GND jumper is solid.
