# capiti-watch streaming demos

Reference card for in-silico and hardware demonstrations of the
streaming classifier. Mirror of the workflow used to validate v0.1.1.

## Quick install

On the test rig (Pi or laptop):

```
pip install -U "capiti==0.1.1"        # add [pi] on the Pi for gpiozero
capiti --version                      # -> capiti 0.1.1
which capiti-watch                    # entry point should be on PATH
```

The bundled ab9 model is the prefix-aug c1 build; threshold schedule
and gate tables ship with it. C and E bundles will follow once their
retrains land.

## Test sequences

Both files used below are reverse-translated from the project's known
proteins so that frame +1 from the first ATG yields the original AA
sequence (no hidden internal stops). Each ends with TAA.

### T1 (positive control, in-set protein)

```
mkdir -p ~/capiti-demo
cat > ~/capiti-demo/make_T1_nt.py <<'PY'
from capiti.seq import CODON_TABLE
inv = {}
for c, a in CODON_TABLE.items():
    if a == "*":
        continue
    inv.setdefault(a, c)
aa = ("GEIRPTIGQQMETGDQRFGDLVFRQLAPNVWQHTSYLDMPGFGAVASNGLIVRDGGRVLVV"
      "DTAWTDDQTAQILNWIKQEINLPVALAVVTHAHQDKMGGMDALHAAGIATYANALSNQLAP"
      "QEGMVAAQHSLTFAANGWVEPATAPNFGPLKVFYPGPGHTSDNITVGIDGTDIAFGGCLIK"
      "DSKAKSLGNLGDADTEHYAASARAFGAAFPKASMIVMSHSAPDSRAAITHTARMADKLR")
print("ATG" + "".join(inv[a] for a in aa) + "TAA")
PY
python ~/capiti-demo/make_T1_nt.py > ~/capiti-demo/T1_nt.txt
wc -c ~/capiti-demo/T1_nt.txt        # expect ~733
```

### Random AA (negative control)

```
cat > ~/capiti-demo/make_random_nt.py <<'PY'
import random
from capiti.seq import CODON_TABLE
random.seed(11)
inv = {}
for c, a in CODON_TABLE.items():
    if a == "*":
        continue
    inv.setdefault(a, c)
aas = "ACDEFGHIKLMNPQRSTVWY"
aa = "".join(random.choice(aas) for _ in range(300))
print("ATG" + "".join(inv[a] for a in aa) + "TAA")
PY
python ~/capiti-demo/make_random_nt.py > ~/capiti-demo/random_nt.txt
wc -c ~/capiti-demo/random_nt.txt    # expect ~906
```

### Optional: FASTA bundle for batch demos

```
cat > ~/capiti-demo/queries.fa <<EOF
>T1_pos
$(cat ~/capiti-demo/T1_nt.txt)
>random_neg
$(cat ~/capiti-demo/random_nt.txt)
EOF
```

## In-silico demos (no hardware)

`--sim-nt` and `--sim-fasta` feed bases through the same scoring loop
the GPIO path uses, so behaviour is identical except for the input
source and the absence of an interrupt pulse.

### Demo 1: positive trigger on T1

```
capiti-watch --sim-nt "$(cat ~/capiti-demo/T1_nt.txt)" --set ab9
```

Expected: trigger near `K=80` AA with `top=T1` and `p_inset>=0.99`.
Last lines of stderr:

```
[K= 80 bases= 240 top=T1 p=1.000 run=5 FIRE]
capiti-watch: TRIGGER at K=80 (top=T1, p_inset=1.000); no pulse (sim or dry run)
capiti-watch[sim]: sim -> K=80 TRIGGERED
```

Exit code: 0.

### Demo 2: negative control (no trigger on random AA)

```
capiti-watch --sim-nt "$(cat ~/capiti-demo/random_nt.txt)" --set ab9 -q
```

Expected: runs through the full input, finishes with `K=300 no
trigger`. Exit code: 1.

### Demo 3: paced for video (10 bases / sec)

Same as demo 1 but slowed so each verdict line scrolls visibly:

```
capiti-watch --sim-nt "$(cat ~/capiti-demo/T1_nt.txt)" --set ab9 \
    --sim-rate-hz 10
```

At 10 Hz a 240-base T1 prefix takes 24 s before triggering -- enough
to narrate over.

### Demo 4: batch (FASTA in, verdicts out)

```
capiti-watch --sim-fasta ~/capiti-demo/queries.fa --set ab9 -q
```

Expected: per-record summary on stderr; `T1_pos -> TRIGGERED`,
`random_neg -> no trigger`.

### Demo 5: tighter threshold (specificity-favoring)

If the default `0.95 / stability=5` is too aggressive for your synth,
demonstrate the conservative profile:

```
capiti-watch --sim-nt "$(cat ~/capiti-demo/T1_nt.txt)" --set ab9 \
    --threshold 0.99 --stability 10 --min-k 80
```

Trigger lands a few residues later but with much lower
single-spike-trigger risk.

## Hardware demos

Hardware path: an external source (synthesizer or a microcontroller
playing back a sequence file) drives the amidite + TET strobe lines on
the Pi. capiti-watch reads those in-process, scores after every codon,
and pulses the abort pin on trigger.

### Wiring (matches capiti-listen / capiti-interrupt)

Inputs (Pi, BCM):

| Pin   | Phys | Role                      |
|-------|------|---------------------------|
| GPIO5 | 29   | A amidite                 |
| GPIO6 | 31   | G amidite                 |
| GPIO13| 33   | C amidite                 |
| GPIO19| 35   | T amidite                 |
| GPIO26| 37   | TET activator strobe      |
| GPIO22| 15   | DONE (rising edge stops)  |

Output:

| Pin   | Phys | Role                                  |
|-------|------|---------------------------------------|
| GPIO17| 11   | Abort pulse (HIGH 100 ms on trigger)  |

Pi GND must be common with the upstream rig. Each line needs an
external pulldown (10 kohm to GND); the internal Pi resistors are
disabled in the driver.

### Demo H1: dry run (visibility only, no abort)

The user feeds the rig a chosen DNA sequence (e.g. T1 NT). Pi reads
pulses, scores live, prints verdicts but does NOT pulse GPIO17.

```
capiti-watch --set ab9 --no-interrupt
```

Watch stderr for the verdict stream as bases come in; expect FIRE on a
real T1 prefix once `K>=50` and `p_inset` holds above 0.95 for 5
consecutive scorings.

Stop with Ctrl-C, the DONE pulse from the rig, or `--max-bases N`.

### Demo H2: positive trigger - real abort on a T1 sequence

User feeds T1 NT to the rig; Pi pulses the abort pin on trigger.

```
capiti-watch --set ab9
```

Expected hardware behaviour:

1. capiti-watch logs progressive `top=T1 p=...` lines as bases arrive.
2. When `p_inset>=0.95` for 5 consecutive scorings (typically near
   K=80 with the v0.1.1 c1 weights), GPIO17 goes HIGH for 100 ms.
3. Process exits 0; the upstream rig observes the abort edge and
   should halt synthesis.

### Demo H3: negative control - feed a non-target sequence

User feeds a benign / random sequence to the rig; capiti-watch should
NOT fire.

```
capiti-watch --set ab9 --max-bases 1000
```

Expected: stream of `top=... p=...` lines with `p_inset` staying low,
no FIRE token, `--max-bases` (or DONE) ends the run, exit code 1, no
abort pulse on GPIO17.

### Demo H4: stricter threshold for production use

If false aborts are unacceptable in the deployment context (and
slightly later detection is OK), tighten the verdict:

```
capiti-watch --set ab9 --threshold 0.99 --stability 10 --min-k 100
```

Same wiring; same trigger behaviour, just less hair-trigger.

### Demo H5: log everything to disk for post-mortem

```
capiti-watch --set ab9 --out /tmp/captured_seq.txt 2> /tmp/verdicts.log
```

`captured_seq.txt` accrues bases as they arrive; `verdicts.log` holds
the full per-codon score trace.

## Variants for non-ab9 sets

Once C and E retrains land in 0.1.2, repeat any of the demos with
`--set C` or `--set E`. Sequences for those targets need to be
reverse-translated from their respective primary fastas (same recipe
as the T1 NT generator above, just point at
`data/C/targets/primary_sequences/T-Cxx.fasta` etc).

## Sanity-check fallbacks

If a demo behaves unexpectedly, in this order:

1. `capiti --version` -> confirm 0.1.1 (or whatever you pushed).
2. `capiti-watch --sim-nt "$(cat ~/capiti-demo/T1_nt.txt)" --set ab9 -q`
   -- if this fails, the issue is software, not hardware.
3. On the Pi, `capiti-listen --max-len 30 -v` to confirm the rig is
   actually delivering pulses end-to-end before re-trying capiti-watch.
4. `capiti-interrupt` on its own (one-shot 100 ms HIGH on GPIO17) to
   confirm the abort line is wired through to the upstream rig.
