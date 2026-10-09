#!/usr/bin/env python3
# halogen-flash-server: source shipped as-is, comments stripped. The README is the documentation.
USAGE = """halogen_tools.py: the text front end of `halogen-tools` for the two modes
that need the tokenizer.

    halogen_tools.py ppl  FILE (--corpus TEXT | --ids IDS.bin) [--worst N] [--json] [binary flags...]
    halogen_tools.py niah FILE (--corpus TEXT | --filler-ids IDS.bin) [--depths 4096,32768]
                              [--positions 0.05,0.5,0.95] [--gen 32] [--out DIR] [--json] [binary flags...]

`ppl` tokenizes the corpus with the mounted tokenizer, runs the binary, and
decodes the `--worst` positions to text. `niah` builds a needle-in-a-haystack
battery from the corpus (the filler), runs it through the binary, and scores
the generations: a case is a hit when the answer string appears in the decoded
output (an id-subsequence match is reported beside it, since a tokenizer need
not re-tokenize an emitted string to the same ids). Every other flag is passed
to the binary as it is.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

BIN = os.environ.get("HALOGEN_TOOLS_BIN", "/usr/local/bin/halogen-tools")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

NEEDLES = [
    {"key": "vault7",
     "fact": "Record 4471: the access code for vault 7 is 84-QUARTZ-19.",
     "lead": "The access code for vault 7 is",
     "answer": "84-QUARTZ-19"},
    {"key": "relay",
     "fact": "Note: the maintenance window for the Kestrel relay is 03:40 UTC on the 22nd.",
     "lead": "The maintenance window for the Kestrel relay is",
     "answer": "03:40 UTC on the 22nd"},
    {"key": "ledger",
     "fact": "Entry: the reconciliation ledger for the Ammet account is filed under case number TR-9083-B.",
     "lead": "The reconciliation ledger for the Ammet account is filed under case number",
     "answer": "TR-9083-B"},
]

def load_tok(path):
    if not os.path.isdir(path):
        sys.exit("tokenizer: %s is not a directory (mount the tokenizer files there, or pass --tokenizer DIR)" % path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)

def read_ids(path):
    b = open(path, "rb").read()
    return list(struct.unpack("<%di" % (len(b) // 4), b))

def write_ids(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<%di" % len(ids), *ids))

def encode_file(tok, path):
    try:
        text = open(path, encoding="utf-8").read()
    except UnicodeDecodeError as e:
        sys.exit("%s is not UTF-8 text (%s); convert it first" % (path, e))
    except OSError as e:
        sys.exit("cannot read %s: %s" % (path, e))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) > 2**31:
        sys.exit("%s tokenizes to %d ids; more than this tool scores in one run" % (path, len(ids)))
    return ids

def cmd_ppl(a, rest):
    tok = load_tok(a.tokenizer)
    tmp = None
    if a.corpus:
        ids = encode_file(tok, a.corpus)
        if len(ids) < 2:
            sys.exit("ppl: %s tokenizes to %d ids; need at least 2" % (a.corpus, len(ids)))
        tmp = tempfile.NamedTemporaryFile(suffix=".ids.bin", delete=False)
        write_ids(tmp.name, ids)
        ids_path = tmp.name
        print("ppl: %s -> %d ids (the mounted tokenizer, no special tokens)" % (a.corpus, len(ids)), file=sys.stderr)
    else:
        ids_path = a.ids
    cmd = [BIN, "ppl", a.file, "--ids", ids_path, "--json"] + rest
    if a.worst:
        cmd += ["--worst", str(a.worst)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    finally:
        if tmp:
            os.unlink(tmp.name)
    if r.returncode != 0:
        sys.exit(r.returncode)
    d = json.loads(r.stdout)

    def decode_worst(obj):
        for w in obj.get("ref", {}).get("worst", []):
            w["ctx_text"] = tok.decode(w["ctx_ids"])
            w["target_text"] = tok.decode([w["target"]])
            w["ref_top_text"] = [[tok.decode([i]), p] for i, p in w["ref_top"]]
            w["this_top_text"] = [[tok.decode([i]), p] for i, p in w["this_top"]]
    for obj in ([d] if "file" in d else [d.get("a", {}), d.get("b", {})]):
        decode_worst(obj)
    if a.json:
        print(json.dumps(d))
        return 0
    for obj in ([d] if "file" in d else [d["a"], d["b"]]):
        print("%s: PPL %.4f (mean NLL %.5f over %d tokens, chunk %d = the %s)" % (
            obj["file"], obj["ppl"], obj["nll"], obj["tokens"], obj["chunk"], obj["path"]))
        if "ref" in obj:
            rf = obj["ref"]
            print("  against %s (top-%d support, a lower bound on the exact KL):" % (rf["path"], rf["topk"]))
            print("  KL(ref || this)  mean %.6f  median %.6f  p90 %.6f  p99 %.6f  max %.6f" % (
                rf["kl"]["mean"], rf["kl"]["median"], rf["kl"]["p90"], rf["kl"]["p99"], rf["kl"]["max"]))
            print("  top-1 agreement  %.3f%%" % (100.0 * rf["top1_agreement"]))
            print("  delta p(target)  mean %+.6f  rms %.6f  |dp| p99 %.6f" % (
                rf["dp"]["mean"], rf["dp"]["rms"], rf["dp"]["abs_p99"]))
            if rf["worst"]:
                print("  the %d positions the two files disagree on most:" % len(rf["worst"]))
            for w in rf["worst"]:
                print("    pos %6d  KL %7.2f  ref p %.3f  this p %.3f  %r -> %r" % (
                    w["pos"], w["kl"], w["ref_p"], w["this_p"], w["ctx_text"][-60:], w["target_text"]))
                print("      ref top: %s   this top: %s" % (
                    ", ".join("%r %.2f" % (t, p) for t, p in w["ref_top_text"]),
                    ", ".join("%r %.2f" % (t, p) for t, p in w["this_top_text"])))
    if "paired" in d:
        p = d["paired"]["paired"]
        print("paired B-A: mean %+.5f nats  t = %+.2f  95%% CI [%+.5f, %+.5f]  PPL ratio %.5f  B better on %d / worse on %d / tied %d" % (
            p["mean_diff"], p["t"] if p["t"] is not None else float("nan"), p["ci95"][0], p["ci95"][1],
            p["ppl_ratio"], p["better"], p["worse"], p["tied"]))
    return 0

def contains(hay, sub):
    return any(hay[i:i + len(sub)] == sub for i in range(len(hay) - len(sub) + 1))

def cmd_niah(a, rest):
    tok = load_tok(a.tokenizer)

    def enc(s):
        return tok(s, add_special_tokens=False)["input_ids"]
    filler = encode_file(tok, a.corpus) if a.corpus else read_ids(a.filler_ids)
    if len(filler) < 256:
        sys.exit("niah: the filler is %d ids; give a longer corpus" % len(filler))
    depths = [int(x) for x in a.depths.split(",")]
    fracs = [float(x) for x in a.positions.split(",")]
    out = a.out or tempfile.mkdtemp(prefix="niah-")
    os.makedirs(out, exist_ok=True)
    cases, rows = [], []
    for nd in NEEDLES:
        fact_ids = enc("\n\n" + nd["fact"] + "\n\n")
        tail_ids = enc("\n\n" + nd["lead"])
        ans_ids = enc(" " + nd["answer"])
        for T in depths:
            room = T - len(fact_ids) - len(tail_ids)
            if room < 64:
                print("niah: skip %s at T=%d, no room for filler" % (nd["key"], T), file=sys.stderr)
                continue
            tiled = room > len(filler)
            fill = (filler * ((room + len(filler) - 1) // len(filler)))[:room] if tiled else filler
            for fr in fracs:
                p = max(0, min(room, int(round(fr * room))))
                ids = fill[:p] + fact_ids + fill[p:room] + tail_ids
                assert len(ids) == T
                name = "%s_T%d_p%02d" % (nd["key"], T, int(round(fr * 100)))
                write_ids(os.path.join(out, name + ".ids"), ids)
                rows.append("%s\t%s\t%d" % (name, os.path.join(out, name + ".ids"), a.gen))
                cases.append({"name": name, "needle": nd["key"], "T": T, "frac": fr, "needle_pos": p,
                              "answer": nd["answer"], "answer_ids": ans_ids, "gen": a.gen, "tiled": tiled})
    manifest = os.path.join(out, "manifest.tsv")
    with open(manifest, "w") as f:
        f.write("\n".join(rows) + "\n")
    json.dump(cases, open(os.path.join(out, "cases.json"), "w"), indent=1)
    print("niah: %d cases at depths %s, positions %s, %d generated tokens each%s -> %s" % (
        len(cases), a.depths, a.positions, a.gen,
        " (the filler is repeated to reach the deeper cases)" if any(c["tiled"] for c in cases) else "", out),
        file=sys.stderr)
    r = subprocess.run([BIN, "niah", a.file, "--manifest", manifest, "--out", out] + rest,
                       stdout=None if not a.json else subprocess.DEVNULL)
    if r.returncode != 0:
        sys.exit(r.returncode)
    scored = []
    for c in cases:
        g = read_ids(os.path.join(out, c["name"] + ".gen"))
        txt = tok.decode(g)
        hit = c["answer"].lower() in txt.lower()
        scored.append(dict(c, text=txt, hit=hit, hit_ids=contains(g, c["answer_ids"])))

    def rate(sel):
        s = [x for x in scored if sel(x)]
        return sum(x["hit"] for x in s), len(s)
    Ts = sorted({c["T"] for c in scored})
    table = []
    for T in Ts:
        row = {"T": T}
        for fr in fracs:
            h, n = rate(lambda x, T=T, fr=fr: x["T"] == T and x["frac"] == fr)
            row["p%02d" % int(round(fr * 100))] = [h, n]
        h, n = rate(lambda x, T=T: x["T"] == T)
        row["all"] = [h, n]
        table.append(row)
    h, n = rate(lambda x: True)
    if a.json:
        print(json.dumps({"tool": "halogen-tools niah", "file": a.file, "depths": depths, "positions": fracs,
                          "gen": a.gen, "by_depth": table, "overall": [h, n],
                          "cases": [{k: v for k, v in c.items() if k != "answer_ids"} for c in scored]}))
        return 0
    print("\nretrieval by depth x needle position (hits / cases)")
    print("  %-8s" % "T" + "".join("  %8s" % ("p%02d" % int(round(fr * 100))) for fr in fracs) + "       all")
    for row in table:
        line = "  %-8d" % row["T"]
        for fr in fracs:
            hh, nn = row["p%02d" % int(round(fr * 100))]
            line += "  %8s" % ("%d/%d" % (hh, nn))
        hh, nn = row["all"]
        line += "   %4d/%-4d %5.1f%%" % (hh, nn, 100.0 * hh / nn if nn else 0.0)
        print(line)
    print("overall %d/%d = %.1f%%" % (h, n, 100.0 * h / n))
    misses = [x for x in scored if not x["hit"]]
    if misses:
        print("misses:")
        for x in misses:
            print("  %-24s want %-22r got %r" % (x["name"], x["answer"], x["text"][:80]))
    return 0

def main():
    ap = argparse.ArgumentParser(description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ppl", "niah"])
    ap.add_argument("file")
    ap.add_argument("--tokenizer", default=os.environ.get("HALOGEN_TOKENIZER", "/tokenizer"))
    ap.add_argument("--corpus", help="a text file; tokenized with the mounted tokenizer")
    ap.add_argument("--ids", help="ppl: pre-tokenized int32 ids instead of --corpus")
    ap.add_argument("--filler-ids", help="niah: pre-tokenized filler instead of --corpus")
    ap.add_argument("--worst", type=int, default=0, help="ppl with --ref: decode the N worst positions")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--depths", default="4096,32768")
    ap.add_argument("--positions", default="0.05,0.5,0.95")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--out", help="niah: where the cases and generations go (a temp dir otherwise); "
                    "ppl: the per-position NLL file, handed to the binary")
    a, rest = ap.parse_known_args()
    if a.mode == "ppl":
        if not (a.corpus or a.ids):
            sys.exit("ppl: --corpus TEXT or --ids IDS.bin")
        if a.out:
            rest = rest + ["--out", a.out]
        return cmd_ppl(a, rest)
    if not (a.corpus or a.filler_ids):
        sys.exit("niah: --corpus TEXT or --filler-ids IDS.bin")
    return cmd_niah(a, rest)

if __name__ == "__main__":
    sys.exit(main())
