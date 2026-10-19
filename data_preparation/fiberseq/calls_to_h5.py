"""Crop per-molecule calls into the candidate windows and write the training H5 files.

Inputs are the ``<chrom>.tsv.zst`` calls from ``fiberbam_to_calls.py`` and the candidate
BED from ``select_regions.py``. Output, in ``--out-dir``:

    active.h5, background.h5   one dataset per window in each of smf_mat, smf_pos,
                               read_id, strand and haplotype, laid out like the mouse
                               ONT files so SMFNet trains on h5_path: [active, background]
    manifest.tsv               every candidate window, its depth, and why it was kept or not

For each window, every molecule overlapping it by at least ``--min-overlap`` bp is cropped
to it. ``smf_mat`` is ``[molecules, positions]`` int8 over the window offsets in
``smf_pos`` that carry at least one call, in the mouse ONT files' encoding: ``1``
unmethylated (protected), ``0`` methylated (m6A, accessible), ``-1`` no call. That is the
reverse of the calls TSVs, where ``1`` is m6A; ``SMF_ENCODING`` is written to the file
attributes.
``strand`` is ``+``/``-`` and ``haplotype`` is ``1``/``2``, or ``0`` when untagged.

A window is kept if at least ``--min-molecules`` molecules span all of it. The background is
then downsampled to ``--background-ratio`` times the active windows, matched on GC in 1%
bins unless ``--no-gc-match``; bins short of background keep all they have. Depth is
filtered first so the ratio holds for the windows actually written.

Every kept window is read back through ``smf_net.data.load_dna`` by its name, and every
observed call must sit on a site ``get_valid_mask(..., "fiberseq_m6a")`` accepts. That
catches a coordinate convention mismatch, which the training loader's assay filter would
otherwise hide by silently dropping the shifted columns.
"""

import argparse
import collections
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch
from calls_io import SUFFIX, iter_calls

from smf_net.data import get_valid_mask, load_dna

ASSAY = "fiberseq_m6a"
# calls characters 0/1/2 (no call / m6A / no m6A) -> smf_mat -1/0/1, matching the mouse ONT
# files, whose builder maps methylated to 0 and unmethylated to 1.
CALL_TO_SMF = np.array([-1, 0, 1], dtype=np.int8)
SMF_ENCODING = "1 unmethylated (protected), 0 methylated (m6A, accessible), -1 no call"
HAPLOTYPE = {"1": 1, "2": 2}
GROUPS = ("smf_mat", "smf_pos", "read_id", "strand", "haplotype")
MANIFEST_COLUMNS = (
    "name",
    "chrom",
    "start",
    "end",
    "class",
    "gc",
    "ccre_id",
    "ccre_class",
    "molecules",
    "full_span",
    "positions",
    "kept",
    "reason",
)


def read_candidates(path):
    """-> {chrom: [window dict]} in file order, and the common window width."""
    by_chrom = collections.defaultdict(list)
    widths = set()
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            start, end = int(f[1]), int(f[2])
            widths.add(end - start)
            by_chrom[f[0]].append(
                {
                    "name": f[3],
                    "chrom": f[0],
                    "start": start,
                    "end": end,
                    "cls": f[4],
                    "gc": float(f[5]),
                    "ccre_id": f[6],
                    "ccre_class": f[7],
                }
            )
    if len(widths) != 1:
        raise SystemExit(f"candidate windows must share one width, got {sorted(widths)}")
    for windows in by_chrom.values():
        windows.sort(key=lambda w: w["start"])
    return by_chrom, widths.pop()


def crop_window(window, molecules, width, min_overlap):
    """-> (smf_mat, smf_pos, read_ids, strands, haplotypes, full_span) for one window."""
    a, b = window["start"], window["end"]
    rows, read_ids, strands, haplotypes, full_span = [], [], [], [], 0
    for start, end, read_id, strand, haplotype, calls in molecules:
        lo, hi = max(a, start), min(b, end)
        if hi - lo < min_overlap:
            continue
        row = np.full(width, -1, dtype=np.int8)
        row[lo - a : hi - a] = CALL_TO_SMF[calls[lo - start : hi - start]]
        rows.append(row)
        read_ids.append(read_id)
        strands.append(strand)
        haplotypes.append(haplotype)
        full_span += start <= a and end >= b
    if not rows:
        return None
    dense = np.stack(rows)
    pos = np.flatnonzero((dense != -1).any(axis=0))
    return (
        dense[:, pos],
        pos.astype(np.int16),
        read_ids,
        strands,
        np.array(haplotypes, dtype=np.int8),
        full_span,
    )


def check_on_valid_sites(path, chrom, fasta, width):
    """Load every window in a per-chromosome file by name and check its calls' sites."""
    with h5py.File(path, "r") as h5:
        names = list(h5["smf_pos"].keys())
        if not names:
            return
        dna = load_dna(fasta, names, region_length=width)
        for name, region_dna in zip(names, dna):
            valid = get_valid_mask(region_dna, ASSAY).numpy()
            pos = h5["smf_pos"][name][:]
            if not valid[pos].all():
                bad = pos[~valid[pos]][:5]
                # Not SystemExit: this runs in a pool worker, and Pool only passes
                # Exception subclasses back, so SystemExit would hang the parent.
                raise ValueError(f"{chrom}: {name} has calls off A/T sites at offsets {bad}")


def process_chrom(job):
    """Crop one chromosome's windows into a temporary H5; -> (chrom, manifest rows, path)."""
    chrom, windows, calls_path, width, args = job
    torch.set_num_threads(1)
    tmp = Path(args["tmp_dir"]) / f"{chrom}.h5"
    rows = iter_calls(calls_path)
    pending = next(rows, None)
    buffer, manifest = [], []
    with h5py.File(tmp, "w") as h5:
        groups = {g: h5.create_group(g) for g in GROUPS}
        for window in windows:
            a, b = window["start"], window["end"]
            # Rows are sorted by start; take every molecule that starts in time to reach
            # min_overlap bases of this window.
            while pending is not None and int(pending[1]) <= b - args["min_overlap"]:
                start, end = int(pending[1]), int(pending[2])
                calls = np.frombuffer(pending[6].encode(), dtype=np.uint8) - ord("0")
                buffer.append(
                    (start, end, pending[3], pending[4], HAPLOTYPE.get(pending[5], 0), calls)
                )
                pending = next(rows, None)
            # Later windows start no earlier, so a molecule that cannot reach this one
            # cannot reach them either.
            buffer = [m for m in buffer if m[1] >= a + args["min_overlap"]]

            cropped = crop_window(window, buffer, width, args["min_overlap"])
            record = dict(window, molecules=0, full_span=0, positions=0)
            if cropped is None or cropped[5] < args["min_molecules"]:
                if cropped is not None:
                    record.update(
                        molecules=len(cropped[2]), full_span=cropped[5], positions=len(cropped[1])
                    )
                record.update(kept=False, reason="low_depth")
                manifest.append(record)
                continue
            mat, pos, read_ids, strands, haplotypes, full_span = cropped
            name = window["name"]
            opts = {"compression": "gzip", "compression_opts": 4}
            groups["smf_mat"].create_dataset(name, data=mat, **opts)
            groups["smf_pos"].create_dataset(name, data=pos, **opts)
            groups["read_id"].create_dataset(
                name, data=np.array(read_ids, dtype=object), dtype=h5py.string_dtype(), **opts
            )
            groups["strand"].create_dataset(name, data=np.array(strands, dtype="S1"), **opts)
            groups["haplotype"].create_dataset(name, data=haplotypes, **opts)
            record.update(
                molecules=len(read_ids),
                full_span=full_span,
                positions=len(pos),
                kept=None,
                reason="",
            )
            manifest.append(record)
        # Drain the stream so a truncated or corrupt file still raises.
        for _ in rows:
            pass
    check_on_valid_sites(tmp, chrom, args["fasta"], width)
    return chrom, manifest, str(tmp)


def select_background(manifest, ratio, gc_match, seed):
    """Mark which depth-passing windows are kept; background downsampled to ratio x active."""
    rng = np.random.default_rng(seed)
    passing = [r for r in manifest if r["kept"] is None]
    active = [r for r in passing if r["cls"] == "active"]
    background = [r for r in passing if r["cls"] == "background"]
    for r in active:
        r.update(kept=True, reason="kept")
    if gc_match:
        bins = lambda rs: np.minimum((np.array([r["gc"] for r in rs]) * 100).astype(int), 99)
        want = np.bincount(bins(active), minlength=100) * ratio if active else np.zeros(100)
        bg_bins = bins(background) if background else np.empty(0, int)
        chosen = set()
        for b in range(100):
            members = np.flatnonzero(bg_bins == b)
            k = min(len(members), round(want[b]))
            chosen.update(rng.choice(members, k, replace=False).tolist())
    else:
        k = min(len(background), round(len(active) * ratio))
        chosen = set(rng.choice(len(background), k, replace=False).tolist())
    for i, r in enumerate(background):
        r.update(kept=i in chosen, reason="kept" if i in chosen else "not_selected")


def write_outputs(manifest, tmp_files, out_dir, attrs):
    out = {cls: h5py.File(out_dir / f"{cls}.h5.tmp", "w") for cls in ("active", "background")}
    try:
        for h5 in out.values():
            for g in GROUPS:
                h5.create_group(g)
        kept = collections.defaultdict(list)
        for r in manifest:
            if r["kept"]:
                kept[r["chrom"]].append(r)
        for chrom, records in kept.items():
            with h5py.File(tmp_files[chrom], "r") as src:
                for r in records:
                    dst = out[r["cls"]]
                    for g in GROUPS:
                        src.copy(src[g][r["name"]], dst[g], name=r["name"])
        for cls, h5 in out.items():
            h5.attrs.update(dict(attrs, window_class=cls))
            h5.close()
            os.replace(out_dir / f"{cls}.h5.tmp", out_dir / f"{cls}.h5")
    except BaseException:
        for cls, h5 in out.items():
            if h5.id.valid:
                h5.close()
            (out_dir / f"{cls}.h5.tmp").unlink(missing_ok=True)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calls-dir", required=True, help=f"directory of <chrom>{SUFFIX} files")
    ap.add_argument("--candidates", required=True, help="BED from select_regions.py")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--min-overlap",
        type=int,
        default=500,
        help="bp a molecule must overlap a window to be stored in it",
    )
    ap.add_argument(
        "--min-molecules",
        type=int,
        default=32,
        help="molecules that must span the whole window for it to be kept",
    )
    ap.add_argument("--background-ratio", type=float, default=1.0)
    ap.add_argument("--no-gc-match", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, width = read_candidates(args.candidates)
    missing = [c for c in candidates if not (Path(args.calls_dir) / f"{c}{SUFFIX}").exists()]
    if missing:
        raise SystemExit(f"no calls file for {', '.join(missing)} in {args.calls_dir}")

    with tempfile.TemporaryDirectory(dir=out_dir, prefix="tmp_") as tmp_dir:
        shared = {
            "min_overlap": args.min_overlap,
            "min_molecules": args.min_molecules,
            "fasta": args.fasta,
            "tmp_dir": tmp_dir,
        }
        jobs = [
            (c, w, str(Path(args.calls_dir) / f"{c}{SUFFIX}"), width, shared)
            for c, w in candidates.items()
        ]
        # Largest chromosomes first so the pool is not left waiting on chr1.
        jobs.sort(key=lambda j: -len(j[1]))
        manifest, tmp_files = [], {}
        with Pool(min(args.workers, len(jobs))) as pool:
            for chrom, rows, path in pool.imap_unordered(process_chrom, jobs):
                manifest += rows
                tmp_files[chrom] = path
                sys.stderr.write(
                    f"  {chrom}: {sum(r['kept'] is None for r in rows)} of "
                    f"{len(rows)} windows pass depth\n"
                )
        order = {c: i for i, c in enumerate(candidates)}
        manifest.sort(key=lambda r: (order[r["chrom"]], r["start"]))
        select_background(manifest, args.background_ratio, not args.no_gc_match, args.seed)
        attrs = {
            "assay_type": ASSAY,
            "smf_encoding": SMF_ENCODING,
            "window_length": width,
            "min_overlap": args.min_overlap,
            "min_molecules": args.min_molecules,
            "background_ratio": args.background_ratio,
            "gc_match": not args.no_gc_match,
            "seed": args.seed,
            "calls_dir": str(Path(args.calls_dir).resolve()),
            "candidates": str(Path(args.candidates).resolve()),
        }
        write_outputs(manifest, tmp_files, out_dir, attrs)

    with open(out_dir / "manifest.tsv", "w") as fh:
        fh.write("\t".join(MANIFEST_COLUMNS) + "\n")
        fh.writelines(
            "\t".join(str(r["cls"] if c == "class" else r[c]) for c in MANIFEST_COLUMNS) + "\n"
            for r in manifest
        )
    counts = collections.Counter((r["cls"], r["reason"]) for r in manifest)
    sys.stderr.write(
        f"active: {counts['active', 'kept']} kept, {counts['active', 'low_depth']} low depth\n"
        f"background: {counts['background', 'kept']} kept, "
        f"{counts['background', 'not_selected']} not selected, "
        f"{counts['background', 'low_depth']} low depth\n"
        f"wrote {out_dir / 'active.h5'}, {out_dir / 'background.h5'}, {out_dir / 'manifest.tsv'}\n"
    )


if __name__ == "__main__":
    main()
