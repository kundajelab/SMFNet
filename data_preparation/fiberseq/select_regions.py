"""Select candidate training windows: registry cCREs split into active and background by ATAC.

Mirrors the mouse ONT regions, which are a CRE bait library split by overlap with ATAC
peaks. Here the library is the ENCODE cCRE registry:

1. Put a ``--width`` window on the centre of every registry cCRE, whatever its class.
2. Drop windows that overlap the blacklist, fall off the chromosome, or are more than
   ``--max-n`` N.
3. Thin greedily in coordinate order so window centres are at least ``--min-spacing``
   apart (with the defaults, at most 50% overlap).
4. A window is **active** if it overlaps an ``--active-peaks`` peak, **background** if it
   overlaps no peak in ``--active-peaks`` or ``--exclude-peaks``, and dropped otherwise
   (it touches only a relaxed peak, so it is neither clearly open nor clearly closed).

No coverage is needed: the depth filter and the GC-matched background downsampling happen
after extraction, in ``calls_to_h5.py``, so that they apply to the windows actually kept.

Output is one sorted BED, ``--out``, for ``fiberbam_to_calls.py --regions``:

    chrom  start  end  name  class  gc  ccre_id  ccre_class

``start``/``end`` are 0-based half-open, ``name`` is the window's H5 key
(``chrom:{start+1}-{end+1}``, a 1-based start with an exclusive end, which
``smf_net.data.load_dna`` reads given the window width), ``class`` is ``active`` or
``background`` and ``gc`` is the window's GC fraction over non-N bases.
"""

import argparse
import collections
import gzip
import re
import sys

import numpy as np
from pyfaidx import Fasta


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_intervals(paths, keep_chrom):
    """-> {chrom: (starts, running max of ends)} for the union of BED files, sorted by start."""
    by_chrom = collections.defaultdict(list)
    for path in paths:
        with open_text(path) as fh:
            for line in fh:
                if line.startswith(("#", "track", "browser")):
                    continue
                f = line.split("\t", 3)
                if len(f) >= 3 and keep_chrom(f[0]):
                    by_chrom[f[0]].append((int(f[1]), int(f[2])))
    out = {}
    for chrom, iv in by_chrom.items():
        a = np.array(sorted(iv), dtype=np.int64)
        out[chrom] = (a[:, 0], np.maximum.accumulate(a[:, 1]))
    return out


def overlaps(starts, ends, intervals):
    """Boolean mask: does [starts, ends) overlap any interval? `intervals` from read_intervals."""
    if intervals is None:
        return np.zeros(len(starts), dtype=bool)
    iv_start, iv_end_max = intervals
    i = np.searchsorted(iv_start, ends, "left") - 1
    hit = i >= 0
    hit[hit] = iv_end_max[i[hit]] > starts[hit]
    return hit


def read_ccres(path, keep_chrom):
    """-> {chrom: [(midpoint, id, class)]} sorted by midpoint."""
    by_chrom = collections.defaultdict(list)
    with open_text(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 10 or not keep_chrom(f[0]):
                continue
            by_chrom[f[0]].append(((int(f[1]) + int(f[2])) // 2, f[3], f[9]))
    for items in by_chrom.values():
        items.sort()
    return by_chrom


def thin(midpoints, min_spacing):
    """Indices kept by a greedy left-to-right pass with centres >= min_spacing apart."""
    keep, last = [], None
    for i, m in enumerate(midpoints):
        if last is None or m - last >= min_spacing:
            keep.append(i)
            last = m
    return np.array(keep, dtype=np.int64)


def chrom_key(chrom):
    """Natural order: chr1 ... chr22, then chrX, chrY and anything else by name."""
    name = chrom.removeprefix("chr")
    return (0, int(name), "") if name.isdigit() else (1, 0, name)


def base_counts(seq):
    """Prefix sums of GC and N, so any window's counts are two lookups."""
    s = np.frombuffer(seq.encode(), dtype=np.uint8)
    gc = np.concatenate([[0], np.cumsum((s == ord("G")) | (s == ord("C")), dtype=np.int32)])
    acgt = np.isin(s, np.frombuffer(b"ACGT", np.uint8))
    n = np.concatenate([[0], np.cumsum(~acgt, dtype=np.int32)])
    return gc, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ccre", required=True, help="per-biosample cCRE registry BED from SCREEN")
    ap.add_argument(
        "--active-peaks",
        required=True,
        nargs="+",
        help="peaks that make a window active, e.g. IDR thresholded ATAC",
    )
    ap.add_argument(
        "--exclude-peaks",
        nargs="*",
        default=[],
        help="further peaks a background window must avoid, e.g. relaxed ATAC",
    )
    ap.add_argument("--blacklist", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chroms", default=r"^chr(\d+|X)$", help="regex on chromosome name")
    ap.add_argument("--width", type=int, default=2114)
    ap.add_argument(
        "--min-spacing",
        type=int,
        default=None,
        help="minimum distance between window centres; default width // 2",
    )
    ap.add_argument("--max-n", type=float, default=0.01, help="maximum fraction of N bases")
    args = ap.parse_args()
    min_spacing = args.width // 2 if args.min_spacing is None else args.min_spacing
    half = args.width // 2

    chrom_re = re.compile(args.chroms)
    keep_chrom = chrom_re.search
    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    ccres = read_ccres(args.ccre, keep_chrom)
    active_iv = read_intervals(args.active_peaks, keep_chrom)
    open_iv = read_intervals(args.active_peaks + args.exclude_peaks, keep_chrom)
    blacklist_iv = read_intervals([args.blacklist], keep_chrom)

    counts = collections.Counter()
    rows = []
    for chrom in sorted(ccres, key=chrom_key):
        if chrom not in fasta:
            raise SystemExit(f"{chrom} from the cCRE registry is not in {args.fasta}")
        items = ccres[chrom]
        mid = np.array([m for m, _, _ in items], dtype=np.int64)
        start = mid - half
        end = start + args.width
        gc_sum, n_sum = base_counts(str(fasta[chrom][:]))
        chrom_len = len(n_sum) - 1
        counts["ccres"] += len(mid)

        ok = (start >= 0) & (end <= chrom_len)
        s_ok, e_ok = np.clip(start, 0, chrom_len), np.clip(end, 0, chrom_len)
        n_frac = (n_sum[e_ok] - n_sum[s_ok]) / args.width
        ok &= ~overlaps(start, end, blacklist_iv.get(chrom))
        ok &= n_frac <= args.max_n
        counts["after_filters"] += int(ok.sum())

        idx = np.nonzero(ok)[0]
        idx = idx[thin(mid[idx], min_spacing)]
        counts["after_thinning"] += len(idx)

        s, e = start[idx], end[idx]
        active = overlaps(s, e, active_iv.get(chrom))
        background = ~overlaps(s, e, open_iv.get(chrom))
        counts["active"] += int(active.sum())
        counts["background"] += int(background.sum())
        counts["ambiguous"] += int((~active & ~background).sum())

        non_n = args.width - (n_sum[e] - n_sum[s])
        gc = (gc_sum[e] - gc_sum[s]) / np.maximum(non_n, 1)
        for j in np.nonzero(active | background)[0]:
            _, ccre_id, ccre_class = items[idx[j]]
            a = int(s[j])
            rows.append(
                f"{chrom}\t{a}\t{a + args.width}\t{chrom}:{a + 1}-{a + args.width + 1}\t"
                f"{'active' if active[j] else 'background'}\t{gc[j]:.4f}\t"
                f"{ccre_id}\t{ccre_class}\n"
            )

    with open(args.out, "w") as out:
        out.writelines(rows)
    sys.stderr.write(
        f"{counts['ccres']} cCREs -> {counts['after_filters']} windows after blacklist/N/edge "
        f"filters -> {counts['after_thinning']} after thinning to centres >= {min_spacing} bp "
        f"apart\n  active {counts['active']}, background {counts['background']}, dropped as "
        f"ambiguous {counts['ambiguous']}; wrote {len(rows)} windows to {args.out}\n"
    )


if __name__ == "__main__":
    main()
