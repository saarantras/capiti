"""Fetch broad-negative (decoy) sequences for each target.

For each Ti:
  1. Read data/targets/active_sites/Ti.json to get UniProt acc.
  2. Query UniProt for its Pfam cross-references -> Pfam family IDs.
  3. Query UniProt for members of those Pfam families (bounded).
  4. Drop anything with >=id_cutoff to the target WT (likely positives).
     (Fast heuristic: length filter + local alignment via sequence-identity
      calculation on the fly, since fetched sets are small.)
  5. Write per-target decoy FASTA to data/decoys/family/Ti.fasta.

Also:
  - Pulls a random SwissProt background sample -> data/decoys/random.fasta.

Writes a summary JSON with counts per target and per source.
"""
import argparse, json, pathlib, sys, time, urllib.request, urllib.parse, random


def fetch_json(url, retries=3):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
            time.sleep(1.5)
    raise last


def fetch_text(url, retries=3):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except Exception as e:
            last = e
            time.sleep(1.5)
    raise last


def pfam_ids_for(accession):
    data = fetch_json(f"https://rest.uniprot.org/uniprotkb/{accession}.json")
    ids = []
    for x in data.get("uniProtKBCrossReferences", []):
        if x.get("database") == "Pfam":
            ids.append(x["id"])
    return ids


def fetch_members_of_pfam(pfam_id, max_n=1000):
    """Query UniProtKB for entries cross-referenced to a Pfam family.
       Returns a list of (accession, name, sequence).
    """
    q = urllib.parse.quote(f'(xref:pfam-{pfam_id}) AND (reviewed:true)')
    url = (f"https://rest.uniprot.org/uniprotkb/search?query={q}"
           f"&format=fasta&size={min(max_n, 500)}")
    out = []
    # UniProt paginates via Link header; for simplicity we loop with cursor.
    cursor_url = url
    seen = 0
    while cursor_url and seen < max_n:
        req = urllib.request.Request(cursor_url)
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            link = r.headers.get("Link", "")
        for name, seq in parse_fasta(body):
            out.append((name, seq))
            seen += 1
            if seen >= max_n:
                break
        # parse next cursor from Link: <url>; rel="next"
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split(";")[0].strip().lstrip("<").rstrip(">")
                break
        cursor_url = nxt
    return out


def parse_fasta(text):
    name, buf = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(buf)
            name = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def read_fasta_file(p):
    return list(parse_fasta(open(p).read()))


def quick_identity(a, b):
    """Rough sequence identity via ungapped alignment at best offset.
       Cheap, good enough for filtering near-duplicates.
    """
    if not a or not b:
        return 0.0
    # align by end-anchored offset sweep bounded to +/- len/4 to keep O(n*k).
    la, lb = len(a), len(b)
    k = min(la, lb)
    best = 0
    window = max(1, k // 4)
    for off in range(-window, window + 1):
        matches = 0
        for i in range(k):
            ai = i + (off if off > 0 else 0)
            bi = i + (-off if off < 0 else 0)
            if ai >= la or bi >= lb:
                break
            if a[ai] == b[bi]:
                matches += 1
        best = max(best, matches)
    return best / k


def fetch_random_background(n=2000):
    """Random-ish sample of reviewed UniProt: use a simple large-page fetch,
       then shuffle-and-trim client-side. Not truly random but good enough.
    """
    url = (f"https://rest.uniprot.org/uniprotkb/search?query=reviewed:true"
           f"&format=fasta&size=500")
    out = []
    cursor_url = url
    while cursor_url and len(out) < n * 5:  # overfetch, then sample
        req = urllib.request.Request(cursor_url)
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            link = r.headers.get("Link", "")
        for name, seq in parse_fasta(body):
            out.append((name, seq))
            if len(out) >= n * 5:
                break
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split(";")[0].strip().lstrip("<").rstrip(">")
                break
        cursor_url = nxt
    random.seed(0)
    random.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-sites", default="data/targets/active_sites")
    ap.add_argument("--primary-seqs", default="data/targets/primary_sequences")
    ap.add_argument("--out", default="data/decoys")
    ap.add_argument("--id-cutoff", type=float, default=0.70)
    ap.add_argument("--max-per-family", type=int, default=600)
    ap.add_argument("--random-n", type=int, default=2000)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "family").mkdir(parents=True, exist_ok=True)

    # load target WT sequences
    wts = {}
    for p in sorted(pathlib.Path(args.primary_seqs).glob("T?.fasta")):
        name, seq = next(parse_fasta(open(p).read()))
        wts[p.stem] = seq

    # cache all target UniProt accessions up front to exclude them as decoys
    target_accs = set()
    for p in pathlib.Path(args.active_sites).glob("*.json"):
        a = json.load(open(p)).get("uniprot")
        if a:
            target_accs.add(a)

    summary = dict(targets={}, random=0)
    for tid, wt in wts.items():
        site_path = pathlib.Path(args.active_sites) / f"{tid}.json"
        if not site_path.exists():
            print(f"[{tid}] no active_sites JSON; skipping")
            continue
        d = json.load(open(site_path))
        acc = d.get("uniprot")
        if not acc:
            print(f"[{tid}] no UniProt accession; skipping")
            continue
        try:
            pfams = pfam_ids_for(acc)
        except Exception as e:
            print(f"[{tid}] pfam lookup failed: {e}")
            pfams = []
        print(f"[{tid}] acc={acc} pfam={pfams}", flush=True)
        collected = {}
        for pf in pfams:
            try:
                mem = fetch_members_of_pfam(pf, max_n=args.max_per_family)
            except Exception as e:
                print(f"  [{pf}] fetch failed: {e}")
                continue
            for name, seq in mem:
                collected[name] = seq
            print(f"  [{pf}] got {len(mem)} (cum={len(collected)})", flush=True)
        # filter out near-WT positives
        filtered = []
        kept_acc = set()
        for name, seq in collected.items():
            acc_candidate = name.split("|")[1] if "|" in name else name
            if acc_candidate in target_accs:
                continue
            ident = quick_identity(seq[:400], wt[:400])
            if ident < args.id_cutoff:
                filtered.append((name, seq))
                kept_acc.add(name)
        print(f"  kept {len(filtered)}/{len(collected)} after id<{args.id_cutoff} filter",
              flush=True)
        (out / "family" / f"{tid}.fasta").write_text(
            "\n".join(f">{n}\n{s}" for n, s in filtered) + ("\n" if filtered else "")
        )
        summary["targets"][tid] = dict(uniprot=acc, pfams=pfams,
                                        n_collected=len(collected),
                                        n_kept=len(filtered))

    # random background
    print("fetching random background...", flush=True)
    try:
        rand = fetch_random_background(args.random_n)
        (out / "random.fasta").write_text(
            "\n".join(f">{n}\n{s}" for n, s in rand) + "\n"
        )
        summary["random"] = len(rand)
        print(f"  random={len(rand)}", flush=True)
    except Exception as e:
        print(f"random fetch failed: {e}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
