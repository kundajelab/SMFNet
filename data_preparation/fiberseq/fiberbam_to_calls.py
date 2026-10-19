"""Extract per-molecule m6A calls from a Fiber-seq BAM into a calls TSV (``.tsv.zst``).

The format is in ``README.md``; output is zstd-compressed (see ``calls_io.py``). One row
per molecule that overlaps any region:

    chrom  start  end  read_id  strand  haplotype  calls

``calls`` has one character per reference base over the molecule's full aligned span:
``0`` no call, ``1`` accessible (m6A), ``2`` protected. Regions only decide *which*
molecules are kept. A molecule is written once at full length even when it spans
several regions, so overlapping windows in the region BED never duplicate data.

Fiber-seq BAMs carry m6A in the ``MM``/``ML`` base-modification tags written by
fibertools, alongside 5mC:

    C+m   5mC on the read's C bases (endogenous CpG methylation, ignored here)
    A+a   m6A on the read's A bases          (Hia5, read strand)
    T-a   m6A opposite the read's T bases    (Hia5, complementary strand)

so the m6A-observable positions are the A and T bases. A reference position is
assayable only when the molecule aligns a base there, that base matches the reference,
and both are A or T. Mismatched bases (heterozygous SNPs, sequencing errors) and CIGAR
D/N gaps are no-calls, so a molecule carrying the non-A/T allele of a SNP never
contributes a "protected" call it could not have made.

For an ``MM`` segment in implicit mode (``.`` or no mode character) every base of that
type not listed is unmodified. In explicit mode (``?``) unlisted bases are unknown and
become no-calls.

Requires ``samtools`` and ``zstd`` on PATH, which the repository's pixi environment
provides. The BAM is streamed through ``samtools view``, so no pysam dependency is needed.
"""

import argparse
import bisect
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from calls_io import COLUMNS, check_path, write_calls
from pyfaidx import Fasta

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
SEG_RE = re.compile(r"^([ACGTUN])([+-])([a-z]+|\d+)([.?])?$")

NO_CALL, ACCESSIBLE, PROTECTED = 0, 1, 2
# unmapped, secondary, QC fail, duplicate, supplementary
EXCLUDE_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
A, T = ord("A"), ord("T")


def revcomp(s):
    return s.translate(COMP)[::-1]


def query_to_ref(cigar, pos):
    """Map query index -> reference coordinate for an alignment starting at `pos`.

    Soft-clipped and inserted query bases map to -1.
    """
    ops = CIGAR_RE.findall(cigar)
    qlen = sum(int(n) for n, op in ops if op in "MIS=X")
    out = np.full(qlen, -1, dtype=np.int64)
    q = r = 0
    for n, op in ops:
        n = int(n)
        if op in "M=X":
            out[q : q + n] = np.arange(r, r + n)
            q += n
            r += n
        elif op in "IS":
            q += n
        elif op in "DN":
            r += n
    return np.where(out >= 0, out + pos, -1)


def m6a_query_calls(seq, is_reverse, mm, ml):
    """Decode the m6A segments of MM/ML into per-query-base arrays, indexed as SEQ.

    Returns ``(prob, known)``: ``prob`` is the ML probability (0-255) for listed bases
    and -1 otherwise; ``known`` is False for bases that an explicit-mode (``?``)
    segment leaves unknown.
    """
    original = revcomp(seq) if is_reverse else seq
    n = len(original)
    arr = np.frombuffer(original.encode(), dtype="S1")
    where = {b: np.flatnonzero(arr == b.encode()) for b in "ACGT"}

    prob = np.full(n, -1, dtype=np.int64)
    known = np.ones(n, dtype=bool)
    ml_i = 0
    for segment in mm.strip(";").split(";"):
        if not segment:
            continue
        head, _, rest = segment.partition(",")
        m = SEG_RE.match(head)
        if not m:
            continue
        base, code, mode = m.group(1), m.group(3), m.group(4)
        deltas = np.array(rest.split(","), dtype=np.int64) if rest else np.empty(0, np.int64)
        probs = ml[ml_i : ml_i + len(deltas)]
        ml_i += len(deltas)
        if code != "a":  # keep m6A; 5mC is a separate analysis
            continue
        idx = where.get(base)
        if idx is None:
            continue
        # each delta skips that many unlisted bases of `base`
        k = np.cumsum(deltas + 1) - 1
        keep = k < len(idx)
        listed = idx[k[keep]]
        if mode == "?":
            known[idx] = False
            known[listed] = True
        np.maximum.at(prob, listed, probs[keep])
    if is_reverse:
        prob, known = prob[::-1], known[::-1]
    return prob, known


class RegionIndex:
    """Sorted regions on one chromosome; may overlap each other."""

    def __init__(self, regions):
        regions.sort()
        self.starts = [r[0] for r in regions]
        self.ends = [r[1] for r in regions]
        self.max_len = max(e - s for s, e in regions)

    def best_overlap(self, start, end):
        """Longest overlap between [start, end) and any single region."""
        lo = bisect.bisect_left(self.starts, start - self.max_len)
        best = 0
        for j in range(lo, bisect.bisect_left(self.starts, end)):
            best = max(best, min(end, self.ends[j]) - max(start, self.starts[j]))
        return best


def read_regions(path, chrom):
    """-> [(start, end)] for one chromosome of a BED."""
    out = []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 3 and f[0] == chrom:
                out.append((int(f[1]), int(f[2])))
    return out


def encode_row(row):
    return (row + ord("0")).astype(np.uint8).tobytes().decode()


def aligned_span(q2r):
    """-> (query indices that align, r_lo, r_hi), or None if nothing aligns."""
    q_idx = np.flatnonzero(q2r >= 0)
    if q_idx.size == 0:
        return None
    return q_idx, int(q2r[q_idx[0]]), int(q2r[q_idx[-1]]) + 1


def molecule_row(seq, q2r, is_reverse, mm, ml, ref, min_ml):
    """-> the calls row over the molecule's aligned reference span."""
    q_idx, r_lo, r_hi = aligned_span(q2r)
    prob, known = m6a_query_calls(seq, is_reverse, mm, ml)

    read_base = np.frombuffer(seq.upper().encode(), dtype=np.uint8)[q_idx]
    ref_pos = q2r[q_idx]
    ref_base = ref[ref_pos]
    assayable = (read_base == ref_base) & ((ref_base == A) | (ref_base == T)) & known[q_idx]

    row = np.zeros(r_hi - r_lo, dtype=np.uint8)
    row[ref_pos[assayable] - r_lo] = PROTECTED
    called = assayable & (prob[q_idx] >= min_ml)
    row[ref_pos[called] - r_lo] = ACCESSIBLE
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bam", required=True)
    ap.add_argument("--regions", required=True, help="region BED; may self-overlap")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--out", required=True, help="output path, ending in .tsv.zst")
    ap.add_argument(
        "--min-ml",
        type=int,
        default=230,
        help="ML cutoff out of 255 for calling a base methylated (~0.9)",
    )
    ap.add_argument("--min-mapq", type=int, default=20)
    ap.add_argument(
        "--min-span",
        type=int,
        default=500,
        help="minimum overlap between a molecule and a single region",
    )
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    check_path(args.out)

    regions = read_regions(args.regions, args.chrom)
    if not regions:
        raise SystemExit(f"no regions on {args.chrom} in {args.regions}")
    index = RegionIndex(list(regions))
    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    ref = np.frombuffer(str(fasta[args.chrom][:]).encode(), dtype=np.uint8)
    sys.stderr.write(
        f"{args.chrom}: {len(regions)} regions, {sum(e - s for s, e in regions) / 1e6:.1f} Mb "
        f"of window bp, chromosome {len(ref) / 1e6:.1f} Mb\n"
    )

    # samtools -L takes a BED, so restrict it to this chromosome.
    sub_bed = tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False)  # noqa: SIM115
    for start, end in regions:
        sub_bed.write(f"{args.chrom}\t{start}\t{end}\n")
    sub_bed.close()

    # samtools stderr goes to a file: a full stderr pipe would block it mid-stream.
    err_log = tempfile.TemporaryFile("w+")  # noqa: SIM115
    label_counts = np.zeros(3, dtype=np.int64)
    n_seen = n_rows = span_total = 0
    cmd = [
        "samtools",
        "view",
        "-@",
        str(args.threads),
        "-M",
        "-q",
        str(args.min_mapq),
        "-F",
        hex(EXCLUDE_FLAGS),
        "-L",
        sub_bed.name,
        args.bam,
        args.chrom,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err_log, text=True, bufsize=1 << 20
        )
        with write_calls(args.out, threads=args.threads) as out:
            out.write("# " + "\t".join(COLUMNS) + "\n")
            out.write(
                f"# fiberbam_to_calls.py bam={args.bam} chrom={args.chrom} "
                f"min_ml={args.min_ml} min_mapq={args.min_mapq} "
                f"exclude_flags={hex(EXCLUDE_FLAGS)} min_span={args.min_span}\n"
            )
            for line in proc.stdout:
                f = line.rstrip("\n").split("\t")
                if len(f) < 11:
                    continue
                read_id, flag, chrom, ref_start, cigar, seq = (
                    f[0],
                    int(f[1]),
                    f[2],
                    int(f[3]) - 1,
                    f[5],
                    f[9],
                )
                if chrom != args.chrom or flag & EXCLUDE_FLAGS or cigar == "*" or seq == "*":
                    continue
                mm = ml = None
                haplotype = "."
                for tag in f[11:]:
                    if tag.startswith("MM:Z:"):
                        mm = tag[5:]
                    elif tag.startswith("ML:B:C"):
                        ml = np.array(tag[7:].split(",") if len(tag) > 7 else [], dtype=np.int64)
                    elif tag.startswith("HP:i:"):
                        haplotype = tag[5:]
                if mm is None or ml is None:
                    continue

                q2r = query_to_ref(cigar, ref_start)
                span = aligned_span(q2r)
                if span is None:
                    continue
                _, r_lo, r_hi = span
                n_seen += 1
                if index.best_overlap(r_lo, r_hi) < args.min_span:
                    continue
                is_reverse = bool(flag & 16)
                row = molecule_row(seq, q2r, is_reverse, mm, ml, ref, args.min_ml)

                out.write(
                    f"{args.chrom}\t{r_lo}\t{r_hi}\t{read_id}\t"
                    f"{'-' if is_reverse else '+'}\t{haplotype}\t{encode_row(row)}\n"
                )
                n_rows += 1
                span_total += r_hi - r_lo
                label_counts += np.bincount(row, minlength=3)

                if n_rows % 100000 == 0:
                    sys.stderr.write(f"  {n_seen} reads seen, {n_rows} written\n")
                    sys.stderr.flush()
            # Checked inside the block so a samtools failure discards the output.
            proc.stdout.close()
            if proc.wait() != 0:
                err_log.seek(0)
                raise SystemExit(
                    f"samtools view failed ({proc.returncode}): {err_log.read().strip()}"
                )
    finally:
        os.unlink(sub_bed.name)
        err_log.close()

    n_at = label_counts[ACCESSIBLE] + label_counts[PROTECTED]
    sys.stderr.write(
        f"wrote {args.out}: {n_seen} reads seen, {n_rows} written, "
        f"mean span {span_total // max(n_rows, 1)} bp, "
        f"{label_counts[NO_CALL]} no-call / {label_counts[ACCESSIBLE]} accessible / "
        f"{label_counts[PROTECTED]} protected, "
        f"m6A rate {label_counts[ACCESSIBLE] / n_at if n_at else 0:.4f}\n"
    )


if __name__ == "__main__":
    main()
