import itertools
import json
import random
from collections import Counter

import torch
from omegaconf import OmegaConf

from smf_net.data import dna_str_to_tensor
from smf_net.evaluation.marginalization import (
    MarginalizationConfig,
    Motif,
    MotifSourceConfig,
    assay_site_counts,
    dinucleotide_shuffle,
    generate_marginalization_design,
    matched_controls,
    read_builtin_motifs,
    read_motifs,
    reverse_complement,
    sample_marginalization,
    select_motifs,
)
from smf_net.evaluation.sampling import SamplingConfig


class FakeModel:
    device = torch.device("cpu")

    def sample(
        self,
        dna: torch.Tensor,
        reads_per_region: int,
        verbose: bool = False,
        sample_steps: int | None = None,
        generator: torch.Generator | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampling_window: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        del verbose, sample_steps, generator, sampling_window
        self.compute_dtypes = {*getattr(self, "compute_dtypes", set()), compute_dtype}
        return dna[:, None, :].remainder(2).expand(-1, reads_per_region, -1)


def test_read_motifs_and_builtin_panel(tmp_path):
    path = tmp_path / "motifs.tsv"
    path.write_text("name\tsequence\tcontrol_sequence\nmotif_a\tACGT\tTGCA\nmotif_b\tCCGG\tGGCC\n")

    motifs = read_motifs(path)

    assert motifs == [
        Motif("motif_a", "ACGT", "TGCA"),
        Motif("motif_b", "CCGG", "GGCC"),
    ]
    assert [motif.name for motif in read_builtin_motifs()][:3] == [
        "CTCF",
        "NRF1",
        "BANP",
    ]


def test_jaspar_panel_controls_match_eligible_sites_in_every_assay():
    motifs = read_builtin_motifs("jaspar_consensus")

    assert {"CTCF", "OCT4::SOX2", "NANOG", "GATA1::TAL1", "MYOD1", "AT-only"} <= {
        motif.name for motif in motifs
    }
    for motif in motifs:
        assert assay_site_counts(motif.control_sequence) == assay_site_counts(motif.sequence)


def test_select_motifs_keeps_requested_order():
    motifs = [Motif("a", "AC", "CA"), Motif("b", "GG", "CC")]

    assert select_motifs(motifs, ("b", "a")) == motifs[::-1]
    assert select_motifs(motifs, None) == motifs


def test_assay_site_counts_use_neutral_flanks():
    # The C of GCG is a CG/GC site, but GCH excludes it for its CpG overlap.
    assert assay_site_counts("GCG") == {
        "cg_gc_accessibility": 1,
        "gch_accessibility": 0,
        "fiberseq_m6a": 0,
    }
    assert assay_site_counts("AGCA")["gch_accessibility"] == 1


def test_dinucleotide_shuffle_preserves_ends_and_dinucleotides():
    sequence = "TTATCTGTGAGGAGCAG"
    rng = random.Random(0)

    def pairs(s):
        return Counter(itertools.pairwise(s))

    shuffles = {dinucleotide_shuffle(sequence, rng) for _ in range(50)}
    assert len(shuffles) > 1
    for shuffled in shuffles:
        assert pairs(shuffled) == pairs(sequence)
        assert (shuffled[0], shuffled[-1]) == (sequence[0], sequence[-1])


def test_matched_control_differs_from_motif_on_both_strands():
    sequence = "GCAGCTGC"  # reverse complement palindrome
    control = next(matched_controls(sequence, seed=1))

    assert control not in {sequence, reverse_complement(sequence)}
    assert sorted(control) == sorted(sequence)
    assert assay_site_counts(control) == assay_site_counts(sequence)


def test_configs_store_hydra_lists_as_tuples():
    marginalization = MarginalizationConfig(base_probs=OmegaConf.create([0.1, 0.2, 0.3, 0.4]))
    motifs = MotifSourceConfig(names=OmegaConf.create(["b", "a"]))

    assert marginalization.base_probs == (0.1, 0.2, 0.3, 0.4)
    assert json.loads(json.dumps(marginalization.base_probs)) == [0.1, 0.2, 0.3, 0.4]
    assert motifs.names == ("b", "a")


def test_generate_marginalization_design_is_paired_and_seeded():
    motifs = [Motif("motif_a", "ACGT", "TGCA")]
    config = MarginalizationConfig(
        n_backgrounds=3,
        n_gc_dinucleotides=2,
        seed=17,
        orientations="forward",
    )

    first = generate_marginalization_design(motifs, 12, config)
    second = generate_marginalization_design(motifs, 12, config)

    assert first.background_dna.shape == (3, 12)
    assert first.motif_dna.shape == (1, 3, 12)
    assert first.control_dna.shape == (1, 3, 12)
    assert first.insertion_starts.tolist() == [4]
    assert torch.equal(first.background_dna, second.background_dna)
    assert torch.equal(first.motif_dna, second.motif_dna)
    assert torch.equal(
        first.motif_dna[0, :, 4:8],
        dna_str_to_tensor("ACGT").expand(3, -1),
    )
    assert torch.equal(
        first.control_dna[0, :, 4:8],
        dna_str_to_tensor("TGCA").expand(3, -1),
    )
    assert torch.equal(
        first.motif_dna[0, :, :4],
        first.background_dna[:, :4],
    )
    assert torch.equal(
        first.control_dna[0, :, 8:],
        first.background_dna[:, 8:],
    )


def test_both_orientations_insert_reverse_complements():
    design = generate_marginalization_design(
        [Motif("m", "AACG", "GTCA")],
        sequence_length=12,
        config=MarginalizationConfig(n_backgrounds=2),
    )

    assert [motif.name for motif in design.motifs] == ["m(+)", "m(-)"]
    assert design.orientations == ("+", "-")
    assert design.motifs[1] == Motif("m(-)", "CGTT", "TGAC")
    assert design.insertion_starts.tolist() == [4, 4]
    assert torch.equal(
        design.motif_dna[1, :, 4:8],
        dna_str_to_tensor("CGTT").expand(2, -1),
    )
    assert torch.equal(
        design.control_dna[1, :, 4:8],
        dna_str_to_tensor("TGAC").expand(2, -1),
    )
    forward = generate_marginalization_design(
        [Motif("m", "AACG", "GTCA")],
        sequence_length=12,
        config=MarginalizationConfig(n_backgrounds=2, orientations="forward"),
    )
    assert [motif.name for motif in forward.motifs] == ["m"]
    assert forward.orientations == ("+",)


def test_background_generation_inserts_exact_non_overlapping_gc_pairs():
    design = generate_marginalization_design(
        [Motif("motif_a", "AA", "TT")],
        sequence_length=12,
        config=MarginalizationConfig(
            n_backgrounds=2,
            n_gc_dinucleotides=6,
            base_probs=(1, 0, 0, 0),
            seed=3,
        ),
    )

    assert torch.equal(
        design.background_dna,
        dna_str_to_tensor("GC" * 6).expand(2, -1),
    )


def test_sample_marginalization_streams_background_motif_and_control():
    design = generate_marginalization_design(
        [Motif("motif_a", "ACGT", "TGCA")],
        sequence_length=12,
        config=MarginalizationConfig(n_backgrounds=3, seed=2, orientations="forward"),
    )
    model = FakeModel()

    batches = list(
        sample_marginalization(
            model,
            design,
            SamplingConfig(
                reads_per_region=2,
                reads_per_chunk=1,
                regions_per_batch=2,
            ),
            compute_dtype=torch.float16,
        )
    )

    assert [batch.condition for batch in batches] == [
        "background",
        "background",
        "motif",
        "motif",
        "control",
        "control",
    ]
    assert batches[0].background_indices == (0, 1)
    assert batches[1].background_indices == (2,)
    assert batches[2].motif_name == "motif_a"
    assert batches[2].insertion_start == 4
    assert batches[2].samples.reads.shape == (2, 2, 12)
    assert torch.equal(batches[2].samples.dna, design.motif_dna[0, :2])
    assert model.compute_dtypes == {torch.float16}
