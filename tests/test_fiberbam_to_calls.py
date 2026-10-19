import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIBERSEQ = Path(__file__).parents[1] / "data_preparation" / "fiberseq"
SCRIPT = FIBERSEQ / "fiberbam_to_calls.py"
sys.path.insert(0, str(FIBERSEQ))
from calls_io import iter_calls

pytestmark = pytest.mark.skipif(
    shutil.which("samtools") is None or shutil.which("zstd") is None,
    reason="needs samtools and zstd on PATH, which the pixi environment provides",
)

# 0-based:      0123456789012345678901234567890123456789
REF = "CCATGATCATGGAATTCGAT" + "C" * 20

# Each read is (name, flag, pos (1-based), mapq, cigar, seq, tags). MM deltas count
# skipped bases of that type in the read's original (sequencer) orientation, which for
# a reverse-strand read is the reverse complement of SEQ.
READS = [
    # Forward, exact match. A at ref 5 called, A at ref 13 below threshold, T at ref 9
    # called from the complementary strand. The leading 5mC segment must be skipped
    # while still consuming its ML value.
    (
        "fwd",
        0,
        1,
        60,
        "20M",
        REF[:20],
        ["MM:Z:C+m.,0;A+a.,1,2;T-a.,2;", "ML:B:C,200,250,100,250", "HP:i:1"],
    ),
    # Reverse strand with a soft clip, a deletion over ref 7-8 (C, A) and two SNPs:
    # ref 10 G read as A (its m6A call is dropped) and ref 12 A read as G (must be a
    # no-call, not protected). Original read is TCCTAATCATAA.
    (
        "rev",
        16,
        3,
        60,
        "2S5M2D5M",
        "TTATGATTAGGA",
        ["MM:Z:C+m?;A+a.,1,0,0;T-a.,0,0,0;", "ML:B:C,250,240,250,50,250,250"],
    ),
    # Explicit-mode A segment: only the listed A (ref 5) is known; other As are
    # no-calls. The implicit T segment still makes every T assayable.
    ("explicit", 0, 1, 60, "20M", REF[:20], ["MM:Z:A+a?,1;T-a.,2;", "ML:B:C,250,250", "HP:i:2"]),
    # Excluded: secondary, duplicate, low MAPQ, and too little overlap with the region.
    ("secondary", 256, 1, 60, "20M", REF[:20], ["MM:Z:A+a.,0;", "ML:B:C,250"]),
    ("duplicate", 1024, 1, 60, "20M", REF[:20], ["MM:Z:A+a.,0;", "ML:B:C,250"]),
    ("lowmapq", 0, 1, 10, "20M", REF[:20], ["MM:Z:A+a.,0;", "ML:B:C,250"]),
    ("short", 0, 18, 60, "5M", REF[17:22], ["MM:Z:A+a.,0;", "ML:B:C,250"]),
]

EXPECTED = {
    "fwd": ("0", "20", "+", "1", "00220120210022220022"),
    "rev": ("2", "14", "-", ".", "210110020002"),
    "explicit": ("0", "20", "+", "2", "00020120010000220002"),
}


def run_converter(tmp_path, bam, out, check=True):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--bam",
            str(bam),
            "--regions",
            str(tmp_path / "regions.bed"),
            "--fasta",
            str(tmp_path / "ref.fa"),
            "--chrom",
            "chrT",
            "--out",
            str(out),
            "--min-span",
            "5",
        ],
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def bam(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(f">chrT\n{REF}\n")
    sam = tmp_path / "reads.sam"
    lines = ["@HD\tVN:1.6\tSO:unsorted", f"@SQ\tSN:chrT\tLN:{len(REF)}"]
    for name, flag, pos, mapq, cigar, seq, tags in READS:
        lines.append(
            "\t".join(
                [
                    name,
                    str(flag),
                    "chrT",
                    str(pos),
                    str(mapq),
                    cigar,
                    "*",
                    "0",
                    "0",
                    seq,
                    "*",
                    *tags,
                ]
            )
        )
    sam.write_text("\n".join(lines) + "\n")
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "sort", "-o", str(bam), str(sam)], check=True)
    subprocess.run(["samtools", "index", str(bam)], check=True)
    (tmp_path / "regions.bed").write_text("chrT\t0\t20\n")
    return bam


@pytest.fixture
def calls(tmp_path, bam):
    out = tmp_path / "chrT.tsv.zst"
    run_converter(tmp_path, bam, out)
    assert not Path(f"{out}.tmp").exists()
    return {row[3]: row for row in iter_calls(out)}


def test_writes_one_row_per_kept_molecule(calls):
    assert set(calls) == set(EXPECTED)


@pytest.mark.parametrize("read_id", sorted(EXPECTED))
def test_calls_match_hand_derived_rows(calls, read_id):
    chrom, start, end, rid, strand, haplotype, row = calls[read_id]
    assert chrom == "chrT" and rid == read_id
    assert (start, end, strand, haplotype, row) == EXPECTED[read_id]
    assert int(end) - int(start) == len(row)


def test_every_call_is_on_a_reference_a_or_t(calls):
    for _, start, _, _, _, _, row in calls.values():
        for offset, call in enumerate(row):
            if call != "0":
                assert REF[int(start) + offset] in "AT"


def test_samtools_failure_leaves_no_output(tmp_path, bam):
    broken = tmp_path / "broken.bam"
    broken.write_bytes(b"not a bam")
    out = tmp_path / "chrT.tsv.zst"
    result = run_converter(tmp_path, broken, out, check=False)
    assert result.returncode != 0
    assert not out.exists() and not Path(f"{out}.tmp").exists()
