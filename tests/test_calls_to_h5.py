import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from smf_net.data import extract_smf_and_pos, filter_smf_regions_for_assay, load_dna

FIBERSEQ = Path(__file__).parents[1] / "data_preparation" / "fiberseq"
SCRIPT = FIBERSEQ / "calls_to_h5.py"
sys.path.insert(0, str(FIBERSEQ))
from calls_io import write_calls
from calls_to_h5 import select_background

pytestmark = pytest.mark.skipif(shutil.which("zstd") is None, reason="needs zstd on PATH")

REF = "ACGTTAGCATGCAATTGCCA" * 3  # 60 bp, GC 0.45 in every 20 bp window
WIDTH = 20
# The calls TSVs use 1 = m6A (accessible), 2 = unmethylated (protected); the h5 files
# reverse that to the ONT encoding, 1 = protected, 0 = accessible.
CALL_TO_SMF = {"0": -1, "1": 0, "2": 1}


def molecule_calls(start, end, ref=REF):
    """Calls string for a molecule: every A/T assayable, every third one accessible."""
    return "".join(
        ("1" if i % 3 == 0 else "2") if ref[i] in "AT" else "0" for i in range(start, end)
    )


# (read_id, start, end, strand, haplotype)
MOLECULES = [
    ("m1", 0, 40, "+", "1"),  # spans window 10-30
    ("m2", 5, 35, "-", "2"),  # spans window 10-30
    ("m3", 25, 50, "+", "."),  # overlaps 10-30 by 5 bp: stored, not full span
    ("m4", 28, 60, "-", "1"),  # overlaps 10-30 by 2 bp: not stored; spans 40-60
]
WINDOWS = [(10, 30, "active"), (40, 60, "background")]


def write_inputs(tmp_path, calls_for=molecule_calls):
    (tmp_path / "ref.fa").write_text(f">chrA\n{REF}\n")
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    with write_calls(calls_dir / "chrA.tsv.zst") as out:
        out.write("# header\n")
        for read_id, start, end, strand, hap in MOLECULES:
            out.write(
                f"chrA\t{start}\t{end}\t{read_id}\t{strand}\t{hap}\t{calls_for(start, end)}\n"
            )
    (tmp_path / "candidates.bed").write_text(
        "".join(
            f"chrA\t{a}\t{b}\tchrA:{a + 1}-{b + 1}\t{cls}\t0.4500\tEH{i}\tPLS\n"
            for i, (a, b, cls) in enumerate(WINDOWS)
        )
    )
    return calls_dir


def run(tmp_path, calls_dir, check=True):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--calls-dir",
            str(calls_dir),
            "--candidates",
            str(tmp_path / "candidates.bed"),
            "--fasta",
            str(tmp_path / "ref.fa"),
            "--out-dir",
            str(tmp_path / "out"),
            "--min-overlap",
            "5",
            "--min-molecules",
            "2",
            "--workers",
            "1",
        ],
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def outputs(tmp_path):
    run(tmp_path, write_inputs(tmp_path))
    return tmp_path / "out"


def test_crops_overlapping_molecules_into_the_window(outputs):
    name = "chrA:11-31"
    with h5py.File(outputs / "active.h5", "r") as h5:
        assert h5.attrs["assay_type"] == "fiberseq_m6a"
        assert [r.decode() for r in h5["read_id"][name][:]] == ["m1", "m2", "m3"]
        assert h5["strand"][name][:].tolist() == [b"+", b"-", b"+"]
        assert h5["haplotype"][name][:].tolist() == [1, 2, 0]
        mat, pos = h5["smf_mat"][name][:], h5["smf_pos"][name][:]
    assert mat.dtype == np.int8
    for row, (_, start, end, _, _) in zip(mat, MOLECULES):
        calls = molecule_calls(start, end)
        for value, offset in zip(row, pos):
            ref_pos = 10 + offset
            inside = start <= ref_pos < end
            assert value == (CALL_TO_SMF[calls[ref_pos - start]] if inside else -1)
    # m3 starts at 25, offset 15 of the window, so it has no calls before that.
    assert (mat[2, pos < 15] == -1).all() and (mat[2, pos >= 15] != -1).any()


def test_windows_without_enough_full_span_molecules_are_dropped(outputs):
    with h5py.File(outputs / "background.h5", "r") as h5:
        assert list(h5["smf_mat"].keys()) == []
    rows = [line.split("\t") for line in (outputs / "manifest.tsv").read_text().splitlines()]
    header, body = rows[0], rows[1:]
    by_name = {r[0]: dict(zip(header, r)) for r in body}
    assert by_name["chrA:11-31"]["reason"] == "kept"
    assert by_name["chrA:11-31"]["full_span"] == "2"
    assert by_name["chrA:41-61"]["reason"] == "low_depth"
    assert by_name["chrA:41-61"]["full_span"] == "1"


def test_output_loads_through_smf_net_without_losing_columns(outputs, tmp_path):
    regions, smf, pos, sources = extract_smf_and_pos(str(outputs / "active.h5"), WIDTH)
    dna = load_dna(str(tmp_path / "ref.fa"), regions, region_length=WIDTH)
    _, kept_smf, kept_pos, _, _ = filter_smf_regions_for_assay(
        regions, smf, pos, dna, sources, "fiberseq_m6a"
    )
    assert [p.tolist() for p in kept_pos] == [p.tolist() for p in pos]
    assert all(torch.equal(a, b) for a, b in zip(kept_smf, smf))


def test_calls_off_a_t_sites_fail_the_check(tmp_path):
    def shifted(start, end):  # calls one base to the right of the A/T sites
        return "0" + molecule_calls(start, end)[:-1]

    result = run(tmp_path, write_inputs(tmp_path, shifted), check=False)
    assert result.returncode != 0 and "off A/T sites" in result.stderr
    assert not (tmp_path / "out" / "active.h5").exists()


def _manifest(active_gc, background_gc):
    return (
        [{"cls": "active", "gc": g, "kept": None} for g in active_gc]
        + [{"cls": "background", "gc": g, "kept": None} for g in background_gc]
        + [{"cls": "background", "gc": 0.40, "kept": False, "reason": "low_depth"}]
    )


def test_background_is_gc_matched_per_one_percent_bin():
    manifest = _manifest([0.401, 0.405, 0.551], [0.402, 0.403, 0.409, 0.552, 0.70])
    select_background(manifest, ratio=1.0, gc_match=True, seed=0)
    kept_bg = sorted(r["gc"] for r in manifest if r["cls"] == "background" and r["kept"])
    assert len(kept_bg) == 3
    assert sum(0.40 <= g < 0.41 for g in kept_bg) == 2 and 0.552 in kept_bg
    assert manifest[-1]["reason"] == "low_depth"


def test_background_without_gc_matching_is_a_uniform_sample():
    manifest = _manifest([0.40, 0.55], [0.40, 0.41, 0.42, 0.70])
    select_background(manifest, ratio=1.5, gc_match=False, seed=0)
    assert sum(r["cls"] == "background" and r["kept"] for r in manifest) == 3
