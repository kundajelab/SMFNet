from types import SimpleNamespace

import torch

from smf_net.data import tensor_to_dna_str
from smf_net.evaluation.regions import (
    SequenceRegion,
    batch_regions,
    regions_from_bed,
    regions_from_fasta,
    regions_from_split,
)


def test_regions_from_fasta_preserves_names_and_integer_encodes(tmp_path):
    fasta_path = tmp_path / "regions.fa"
    fasta_path.write_text(">first\nACGT\n>second\nTNCA\n")

    region_source = regions_from_fasta(fasta_path)
    assert region_source.total_regions == 2
    regions = list(region_source)

    assert [region.region_id for region in regions] == ["first", "second"]
    assert [tensor_to_dna_str(region.dna) for region in regions] == [
        "ACGT",
        "TNCA",
    ]
    assert all(region.dna.dtype == torch.int8 for region in regions)
    assert all(region.source.kind == "fasta" for region in regions)
    assert all(region.source.path == str(fasta_path.resolve()) for region in regions)


def test_regions_from_fasta_center_resizes_sequences(tmp_path):
    fasta_path = tmp_path / "regions.fa"
    fasta_path.write_text(">long\nAACCGG\n>short\nCG\n")

    regions = list(regions_from_fasta(fasta_path, sequence_length=4))

    assert [tensor_to_dna_str(region.dna) for region in regions] == [
        "ACCG",
        "NCGN",
    ]


def test_regions_from_bed_resizes_and_respects_strand(tmp_path):
    reference_path = tmp_path / "reference.fa"
    reference_path.write_text(">chr1\nAACCGGTTAACC\n")
    bed_path = tmp_path / "regions.bed"
    bed_path.write_text("chr1\t1\t5\tplus\t0\t+\nchr1\t4\t6\tminus\t0\t-\n")

    region_source = regions_from_bed(bed_path, reference_path, sequence_length=4)
    assert region_source.total_regions == 2
    regions = list(region_source)

    assert [region.region_id for region in regions] == [
        "chr1:1-5",
        "chr1:3-7",
    ]
    assert [tensor_to_dna_str(region.dna) for region in regions] == [
        "ACCG",
        "ACCG",
    ]
    assert all(region.source.kind == "bed" for region in regions)
    assert all(region.source.path == str(bed_path.resolve()) for region in regions)
    assert all(
        region.source.reference_fasta_path == str(reference_path.resolve()) for region in regions
    )


def test_batch_regions_streams_named_stacks():
    regions = [
        SequenceRegion("a", torch.tensor([0, 1], dtype=torch.int8)),
        SequenceRegion("b", torch.tensor([2, 3], dtype=torch.int8)),
        SequenceRegion("c", torch.tensor([3, 2], dtype=torch.int8)),
    ]

    batches = list(batch_regions(regions, batch_size=2))

    assert [batch.region_ids for batch in batches] == [("a", "b"), ("c",)]
    assert torch.equal(
        batches[0].dna,
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int8),
    )


def test_regions_from_split_uses_dataset_region_iteration():
    observed = [
        SimpleNamespace(
            region_id="chr1:2-6",
            dna=torch.tensor([0, 1, 2, 3], dtype=torch.int8),
            source=None,
        )
    ]

    class FakeDataset:
        def iter_regions(self):
            yield from observed

    class FakeDataModule:
        val_dataset = FakeDataset()

        def __init__(self):
            self.stages = []

        def setup(self, stage):
            self.stages.append(stage)

    data = FakeDataModule()

    regions = list(regions_from_split(data, "val"))

    assert data.stages == ["validate"]
    assert regions[0].region_id == "chr1:2-6"
    assert torch.equal(regions[0].dna, observed[0].dna)
