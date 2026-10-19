import subprocess
import sys
from pathlib import Path

import pytest

from smf_net.data import load_dna, tensor_to_dna_str

SCRIPT = Path(__file__).parents[1] / "data_preparation" / "fiberseq" / "select_regions.py"
WIDTH, SPACING = 200, 100


def chr2_sequence():
    seq = list("AT" * 10000)  # 20 kb, GC 0
    seq[4900:4950] = "GC" * 25  # window [4900, 5100) has GC 50/200
    seq[15000:15500] = "N" * 500
    return "".join(seq)


# (chrom, cCRE midpoint, class); each cCRE is 100 bp centred on the midpoint
CCRES = [
    ("chr2", 1000, "PLS"),  # overlaps an IDR peak -> active
    ("chr2", 1050, "pELS"),  # centre 50 bp from the previous one -> thinned
    ("chr2", 3000, "dELS"),  # overlaps only a relaxed peak -> ambiguous, dropped
    ("chr2", 5000, "Low-DNase"),  # no peaks -> background
    ("chr2", 6950, "Low-DNase"),  # blacklisted -> dropped before thinning, so...
    ("chr2", 7000, "Low-DNase"),  # ...this one, 50 bp later, survives as background
    ("chr2", 15200, "Low-DNase"),  # all N -> dropped
    ("chr2", 50, "Low-DNase"),  # window runs off the chromosome start -> dropped
    ("chr10", 500, "Low-DNase"),  # background; written after chr2 (natural order)
    ("chrUn_x", 500, "Low-DNase"),  # excluded by the default chromosome regex
]

EXPECTED = [
    ("chr2", 900, 1100, "active", 0.0),
    ("chr2", 4900, 5100, "background", 0.25),
    ("chr2", 6900, 7100, "background", 0.0),
    ("chr10", 400, 600, "background", 0.0),
]


@pytest.fixture
def selected(tmp_path):
    seqs = {"chr2": chr2_sequence(), "chr10": "AT" * 500, "chrUn_x": "AT" * 500}
    fasta = tmp_path / "ref.fa"
    fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in seqs.items()))
    ccre = tmp_path / "ccre.bed"
    ccre.write_text(
        "".join(
            f"{c}\t{m - 50}\t{m + 50}\tEH{i}\t0\t.\t{m - 50}\t{m + 50}\t0,0,0\t{cls}\tAll-data\n"
            for i, (c, m, cls) in enumerate(CCRES)
        )
    )
    idr = tmp_path / "idr.bed"
    idr.write_text("chr2\t1050\t1060\n")
    relaxed = tmp_path / "relaxed.bed"
    relaxed.write_text("chr2\t1050\t1060\nchr2\t3090\t3200\n")
    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_text("chr2\t6850\t6890\tHigh Signal Region\n")
    out = tmp_path / "candidates.bed"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ccre",
            str(ccre),
            "--active-peaks",
            str(idr),
            "--exclude-peaks",
            str(relaxed),
            "--blacklist",
            str(blacklist),
            "--fasta",
            str(fasta),
            "--out",
            str(out),
            "--width",
            str(WIDTH),
            "--min-spacing",
            str(SPACING),
        ],
        check=True,
        capture_output=True,
    )
    return fasta, [line.rstrip("\n").split("\t") for line in out.read_text().splitlines()]


def test_selects_and_classifies_windows(selected):
    _, rows = selected
    got = [(r[0], int(r[1]), int(r[2]), r[4], float(r[5])) for r in rows]
    assert got == EXPECTED


def test_names_load_the_window_with_default_loader_settings(selected):
    fasta, rows = selected
    dna = load_dna(str(fasta), [r[3] for r in rows], region_length=WIDTH)
    seqs = {}
    name = None
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:]
        else:
            seqs[name] = line
    for d, r in zip(dna, rows):
        assert tensor_to_dna_str(d) == seqs[r[0]][int(r[1]) : int(r[2])]
