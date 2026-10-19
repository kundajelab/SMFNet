"""Build the ``jaspar_consensus`` marginalization motif panel from JASPAR.

This generates the panel shipped as
``src/smf_net/evaluation/motifs/jaspar_consensus.tsv``. It only needs rerunning
to change the panel; the output is deterministic for a given seed, so a rerun
reproduces the shipped rows.

Each motif is the consensus of its JASPAR matrix, the highest-count base at
every position. Its control is, among seeded shuffles with the same number of
eligible sites as the motif in every assay (see ``matched_controls``), the one
scoring lowest against the motif's own matrix, on either strand and at any
offset of up to half the motif length. The composition checks are written as
they are; each is already site-matched to its control.

Example::

    python data_preparation/motif_panel/build_jaspar_panel.py \\
      --output src/smf_net/evaluation/motifs/jaspar_consensus.tsv
"""

import argparse
import csv
import json
import math
from itertools import islice
from urllib.request import urlopen

from smf_net.evaluation.marginalization import matched_controls, reverse_complement

JASPAR_API = "https://jaspar.elixir.no/api/v1/matrix/{}/?format=json"

# (panel name, JASPAR matrix ID, group). SOX3 (MA0514.3) is left out: its
# consensus is identical to SOX2's. GATA1 uses MA0035.3 because the consensus
# of the latest version (MA0035.5, CTAATCT) lacks the GATA core.
PANEL = (
    ("CTCF", "MA0139.2", "general"),
    ("NRF1", "MA0506.3", "general"),
    ("BANP", "MA2503.1", "general"),
    ("OCT4::SOX2", "MA0142.1", "mESC"),
    ("ESRRB", "MA0141.4", "mESC"),
    ("KLF4", "MA0039.5", "mESC"),
    ("SOX2", "MA0143.5", "mESC"),
    ("NFYA", "MA0060.4", "mESC"),
    ("ZIC3", "MA0697.3", "mESC"),
    ("NANOG", "MA2339.1", "mESC"),
    ("REST", "MA0138.3", "mESC"),
    ("PAX6", "MA0069.1", "NPC"),
    ("ASCL1", "MA1100.3", "NPC"),
    ("NEUROG2", "MA0669.1", "NPC"),
    ("RFX3", "MA0798.3", "NPC"),
    ("GATA1", "MA0035.3", "MEL"),
    ("GATA1::TAL1", "MA0140.3", "MEL"),
    ("MAF::NFE2", "MA0501.2", "MEL"),
    ("KLF1", "MA0493.3", "MEL"),
    ("MYOD1", "MA0499.3", "C2C12"),
    ("MEF2C", "MA0497.2", "C2C12"),
)

# (name, sequence, control): base-composition checks with no JASPAR matrix.
COMPOSITION = (
    ("AT-only", "AAAAAAAA", "TTTTTTTT"),
    ("GC-only", "GGGGGGGG", "CCCCCCCC"),
)


def fetch_pfm(matrix_id: str) -> dict[str, list[float]]:
    with urlopen(JASPAR_API.format(matrix_id), timeout=30) as response:
        return json.load(response)["pfm"]


def consensus(pfm: dict[str, list[float]]) -> str:
    return "".join(
        max("ACGT", key=lambda base: pfm[base][position]) for position in range(len(pfm["A"]))
    )


def log_odds(pfm: dict[str, list[float]]) -> list[dict[str, float]]:
    """Per-position log2 odds against a uniform background, pseudocount 0.8."""

    columns = []
    for position in range(len(pfm["A"])):
        total = sum(pfm[base][position] for base in "ACGT") + 0.8
        columns.append(
            {base: math.log2((pfm[base][position] + 0.2) / total / 0.25) for base in "ACGT"}
        )
    return columns


def best_score(sequence: str, pwm: list[dict[str, float]]) -> float:
    """Best matrix score over both strands and offsets up to half its length.

    Positions falling outside the sequence score 0, as for a uniform base.
    """

    reach = len(pwm) // 2
    return max(
        sum(
            pwm[position][strand[position + offset]]
            for position in range(len(pwm))
            if 0 <= position + offset < len(strand)
        )
        for strand in (sequence, reverse_complement(sequence))
        for offset in range(-reach, reach + 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidates",
        type=int,
        default=2000,
        help="Site-matched shuffles scored per motif",
    )
    args = parser.parse_args()

    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "sequence", "control_sequence", "jaspar_id", "group"])
        print("name         matrix    consensus          control            score: motif / control")
        for name, matrix_id, group in PANEL:
            pfm = fetch_pfm(matrix_id)
            pwm = log_odds(pfm)
            sequence = consensus(pfm)
            candidates = list(islice(matched_controls(sequence, seed=args.seed), args.candidates))
            if not candidates:
                raise ValueError(f"No site-matched control found for {name}")
            control = min(candidates, key=lambda c: best_score(c, pwm))
            writer.writerow([name, sequence, control, matrix_id, group])
            print(
                f"{name:12s} {matrix_id}  {sequence:18s} {control:18s} "
                f"{best_score(sequence, pwm):5.1f} / {best_score(control, pwm):5.1f}"
            )
        for name, sequence, control in COMPOSITION:
            writer.writerow([name, sequence, control, "", "composition"])


if __name__ == "__main__":
    main()
