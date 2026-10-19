from itertools import product

import h5py
import pytest
import torch

from smf_net.data import (
    ASSAYS,
    RegionSource,
    SMFItem,
    SparseDataModuleConfig,
    SparseSMFDataModule,
    SparseSMFDataset,
    bernoulli_thin_reads,
    densify_smf,
    dna_str_to_tensor,
    get_valid_mask,
    load_dna,
    reverse_complement_dna_and_smf,
    single_threaded_torch,
)


def _build_dataset(
    reads_per_region: int = 2,
    region_len: int = 12,
    model_input_length: int = 8,
) -> SparseSMFDataset:
    smf_pos = torch.tensor([0, 2, 4, 6, 8, 10], dtype=torch.long)
    num_reads_total = 4
    smf_matrix = torch.randint(0, 2, (num_reads_total, smf_pos.numel()), dtype=torch.int8)
    dna_tensor = dna_str_to_tensor("CG" * (region_len // 2)).unsqueeze(0)
    return SparseSMFDataset(
        smf_list=[smf_matrix],
        pos_list=[smf_pos],
        dna=dna_tensor,
        regions=["chr1_1-12"],
        reads_per_region=reads_per_region,
        train=False,
        model_input_length=model_input_length,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_cg_gc_accessibility_mask_includes_cpg_and_gpc_cytosines(batched):
    dna = dna_str_to_tensor("CGCGCNACGT")
    expected = torch.zeros_like(dna, dtype=torch.bool)
    expected[[0, 2, 4, 7]] = True
    if batched:
        dna = torch.stack([dna, dna])
        expected = torch.stack([expected, expected])

    mask = get_valid_mask(dna, "cg_gc_accessibility")

    assert torch.equal(mask, expected)
    assert mask.shape == dna.shape
    assert mask.dtype == torch.bool
    assert mask.device == dna.device


def test_dataset_filters_ineligible_and_unobserved_columns():
    smf = torch.tensor([[0, 1, 0, -1, 1], [1, 0, 1, -1, 0]], dtype=torch.int8)
    dataset = SparseSMFDataset(
        [smf],
        [torch.tensor([2, 4, 8, 11, 15])],
        dna_str_to_tensor("AGCACGAGCGGCTAGCCA").unsqueeze(0),
        model_input_length=18,
        assay_type="gch_accessibility",
    )

    assert torch.equal(dataset.pos_list[0], torch.tensor([2, 15]))
    assert torch.equal(dataset.smf[0], smf[:, [0, 4]])


def test_cg_gc_accessibility_supports_explicit_rc():
    dataset = SparseSMFDataset(
        [torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int8)],
        [torch.tensor([1, 3, 5])],
        dna_str_to_tensor("ACGCGCAT").unsqueeze(0),
        reads_per_region=1,
        model_input_length=6,
    )
    read_idx = torch.tensor([1])
    expected = dataset.get_item(0, read_idx, start=0)
    explicit = dataset.get_item(0, read_idx, start=0, reverse_complement=True)
    dna_rc, smf_rc = reverse_complement_dna_and_smf(expected.dna, expected.smf)
    assert torch.equal(explicit.dna, dna_rc)
    assert torch.equal(explicit.smf, smf_rc)


# Contains GCH, CGC, GCG and CpG contexts, with sites near both ends.
MIXED_CONTEXT_DNA = "AGCACGCAGCGAGCATGCTCGA"


def _fully_observed_dataset(
    assay_type: str, num_reads: int = 2, model_input_length: int = 16, **kwargs
) -> tuple[SparseSMFDataset, torch.Tensor]:
    """Dataset with a call at every eligible site, plus the dense calls."""
    dna = dna_str_to_tensor(MIXED_CONTEXT_DNA)
    positions = torch.nonzero(get_valid_mask(dna, assay_type)).flatten()
    smf = (torch.arange(num_reads * positions.numel()).reshape(num_reads, -1) % 2).to(torch.int8)
    dataset = SparseSMFDataset(
        [smf],
        [positions],
        dna.unsqueeze(0),
        model_input_length=model_input_length,
        assay_type=assay_type,
        **kwargs,
    )
    return dataset, densify_smf(smf, positions, dna.numel())


def test_rc_augmentation_defaults_to_assay_and_can_be_overridden():
    assert ASSAYS["cg_gc_accessibility"].rc_augmentation
    assert not ASSAYS["gch_accessibility"].rc_augmentation
    assert ASSAYS["fiberseq_m6a"].rc_augmentation
    for assay_type in ASSAYS:
        default, _ = _fully_observed_dataset(assay_type)
        assert default.rc_augmentation == ASSAYS[assay_type].rc_augmentation
        for override in (False, True):
            dataset, _ = _fully_observed_dataset(assay_type, rc_augmentation=override)
            assert dataset.rc_augmentation is override


@pytest.mark.parametrize("assay_type", sorted(ASSAYS))
@pytest.mark.parametrize("rc_augmentation", [None, False, True])
@pytest.mark.parametrize("rc_draw", [0.0, 1.0])
def test_training_item_matches_drawn_jitter_rc_and_reads(
    monkeypatch, assay_type, rc_augmentation, rc_draw
):
    dataset, _ = _fully_observed_dataset(
        assay_type, reads_per_region=1, train=True, rc_augmentation=rc_augmentation
    )
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([1]))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([rc_draw]))
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: torch.tensor([1]))
    rc = dataset.rc_augmentation and rc_draw < 0.5
    expected = dataset.get_item(0, torch.tensor([1]), 1, reverse_complement=rc)
    item = dataset[0]
    assert torch.equal(item.dna, expected.dna)
    assert torch.equal(item.smf, expected.smf)
    assert (item.smf != -1).any()


def test_gch_accessibility_mask_matches_dgch_in_every_context():
    # All four-base contexts, including unknown flanks and CpG overlaps.
    dna = torch.tensor(list(product(range(-1, 4), repeat=4)), dtype=torch.int8)
    expected = torch.zeros(dna.shape, dtype=torch.bool)
    expected[:, 2] = (
        (dna[:, 0] >= 0)
        & (dna[:, 0] != 1)
        & (dna[:, 1] == 2)
        & (dna[:, 2] == 1)
        & (dna[:, 3] >= 0)
        & (dna[:, 3] != 2)
    )
    assert torch.equal(get_valid_mask(dna, "gch_accessibility"), expected)


@pytest.mark.parametrize("assay_type", sorted(ASSAYS))
def test_rc_preserves_calls_and_round_trips(assay_type):
    # All four-base contexts, including unknown flanks and CpG overlaps.
    dna = torch.tensor(list(product(range(-1, 4), repeat=4)), dtype=torch.int8)
    valid = get_valid_mask(dna, assay_type)
    smf = torch.full_like(dna, -1)
    smf[valid] = torch.arange(valid.sum(), dtype=torch.int8) % 2
    # RC keeps one call for the two cytosines of a CGC, so give them the same
    # call, as merged reverse-strand reads do.
    cgc = (dna[:, :-2] == 1) & (dna[:, 1:-1] == 2) & (dna[:, 2:] == 1)
    rows, cols = torch.nonzero(cgc & valid[:, 2:], as_tuple=True)
    smf[rows, cols + 2] = smf[rows, cols]
    for sequence, calls in zip(dna, smf, strict=True):
        calls = calls.unsqueeze(0)
        dna_rc, smf_rc = reverse_complement_dna_and_smf(sequence, calls, assay_type)
        assert torch.equal(smf_rc[0] != -1, get_valid_mask(dna_rc, assay_type))
        restored_dna, restored_smf = reverse_complement_dna_and_smf(dna_rc, smf_rc, assay_type)
        assert torch.equal(restored_dna, sequence)
        assert torch.equal(restored_smf, calls)


def test_fiberseq_m6a_mask_is_every_known_a_and_t():
    dna = dna_str_to_tensor("ACGTNAT")
    assert get_valid_mask(dna, "fiberseq_m6a").tolist() == [
        True,
        False,
        False,
        True,
        False,
        True,
        True,
    ]


def test_fiberseq_m6a_rc_mirrors_each_call_without_shifting_it():
    # An m6A at a top-strand A becomes one opposite the mirrored T.
    dna = dna_str_to_tensor("AACGT")
    smf = torch.tensor([[1, 0, -1, -1, 1], [0, -1, -1, -1, 0]], dtype=torch.int8)
    dna_rc, smf_rc = reverse_complement_dna_and_smf(dna, smf, "fiberseq_m6a")
    assert dna_rc.tolist() == dna_str_to_tensor("ACGTT").tolist()
    assert smf_rc.tolist() == [[1, -1, -1, 0, 1], [0, -1, -1, -1, 0]]


def test_cg_gc_rc_merges_cgc_preferring_gpc_and_falling_back_per_read():
    dna = dna_str_to_tensor("ACGCA")
    # Columns 1 and 3 are the CpG and GpC cytosines of the CGC.
    smf = torch.tensor(
        [
            [-1, 1, -1, -1, -1],  # GpC missing: fall back to CpG
            [-1, 1, -1, 0, -1],  # both observed: GpC wins
            [-1, -1, -1, 0, -1],  # CpG missing: GpC
            [-1, -1, -1, -1, -1],  # neither observed
        ],
        dtype=torch.int8,
    )
    dna_rc, smf_rc = reverse_complement_dna_and_smf(dna, smf)
    assert dna_rc.tolist() == dna_str_to_tensor("TGCGT").tolist()
    assert smf_rc[:, 2].tolist() == [1, 0, 0, -1]
    assert (smf_rc[:, [0, 1, 3, 4]] == -1).all()


def test_cg_gc_rc_copies_gcg_call_to_both_cgc_cytosines():
    dna = dna_str_to_tensor("AGCGA")
    smf = torch.tensor([[-1, -1, 1, -1, -1], [-1, -1, 0, -1, -1]], dtype=torch.int8)
    dna_rc, smf_rc = reverse_complement_dna_and_smf(dna, smf)
    assert dna_rc.tolist() == dna_str_to_tensor("TCGCT").tolist()
    assert smf_rc.tolist() == [[-1, 1, -1, 1, -1], [-1, 0, -1, 0, -1]]


@pytest.mark.parametrize("assay_type", sorted(ASSAYS))
@pytest.mark.parametrize("start", range(len(MIXED_CONTEXT_DNA) - 16 + 1))
def test_cropped_region_keeps_exactly_the_sites_eligible_in_the_window(assay_type, start):
    # Cropping can remove a flank that makes a site eligible.
    dataset, dense = _fully_observed_dataset(assay_type)
    region = dataset.get_region(0, start=start)
    eligible = torch.nonzero(get_valid_mask(region.dna, assay_type)).flatten()
    assert torch.equal(region.positions, eligible)
    assert torch.equal(region.smf, dense[:, start + eligible])


@pytest.mark.parametrize("assay_type", sorted(ASSAYS))
@pytest.mark.parametrize("start", range(len(MIXED_CONTEXT_DNA) - 16 + 1))
@pytest.mark.parametrize("reverse_complement", [False, True])
def test_item_calls_land_exactly_on_eligible_sites(assay_type, start, reverse_complement):
    dataset, _ = _fully_observed_dataset(assay_type)
    item = dataset.get_item(0, torch.tensor([0, 1]), start, reverse_complement=reverse_complement)
    eligible = get_valid_mask(item.dna, assay_type)
    assert torch.equal(item.smf != -1, eligible.expand_as(item.smf))


def test_bernoulli_thinning_discards_regions_with_no_retained_reads():
    regions = ["chr1_1-4", "chr1_5-8"]
    mats = [torch.zeros((1, 2), dtype=torch.int8) for _ in regions]
    positions = [torch.arange(2) for _ in regions]
    sources = [RegionSource("observed_h5", "unused.h5") for _ in regions]

    thinned = bernoulli_thin_reads(
        regions,
        mats,
        positions,
        sources,
        keep_probability=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    thinned_regions, thinned_mats, thinned_positions, thinned_sources = thinned
    assert thinned_regions == [regions[0]]
    assert len(thinned_mats) == len(thinned_positions) == len(thinned_sources) == 1


def test_sparse_dataset_get_item_shapes_and_values():
    reads_per_region = 2
    dataset = _build_dataset(reads_per_region=reads_per_region)

    item = dataset[0]

    assert item.smf.shape == (
        reads_per_region,
        dataset.model_input_length,
    )
    # make sure densify inserted -1 in masked positions
    assert torch.any(item.smf == -1)

    assert item.dna.shape == (dataset.model_input_length,)
    assert item.dna.dtype == torch.int8


def test_sparse_dataset_get_region_keeps_all_reads_and_sparse_positions():
    dataset = _build_dataset(reads_per_region=2)

    region = dataset.get_region(0)

    assert region.region_id == "chr1_1-12"
    assert region.window_start == 2
    assert region.dna.shape == (8,)
    assert region.smf.shape[0] == 4
    assert torch.equal(region.positions, torch.tensor([0, 2, 4, 6]))

    cropped = region.crop(slice(2, 6))
    assert cropped.window_start == 4
    assert cropped.dna.shape == (4,)
    assert torch.equal(cropped.positions, torch.tensor([0, 2]))
    assert cropped.smf.shape == (4, 2)


def test_sparse_dataset_collate_fn_outputs_one_hot_dna():
    reads_per_region = 3
    dataset = _build_dataset(reads_per_region=reads_per_region)

    batch_items: list[SMFItem] = [dataset[0], dataset[0]]
    batched = SparseSMFDataset.collate_fn(batch_items)

    assert batched.smf.shape == (
        len(batch_items),
        reads_per_region,
        dataset.model_input_length,
    )
    assert batched.dna.shape == (
        len(batch_items),
        4,
        dataset.model_input_length,
    )
    # each column should be one-hot encoded
    assert torch.allclose(
        batched.dna.sum(dim=1),
        torch.ones(len(batch_items), dataset.model_input_length),
    )


def test_data_module_drops_regions_without_eligible_observations(monkeypatch):
    cfg = SparseDataModuleConfig(
        fasta_path="unused.fa",
        h5_path="unused.h5",
        val_chrm=["chr1"],
        h5_region_length=8,
        model_input_length=8,
        assay_type="gch_accessibility",
        num_workers=0,
        rc_augmentation=True,
    )
    data_module = SparseSMFDataModule(cfg)
    source = RegionSource("observed_h5", "unused.h5")
    monkeypatch.setattr(
        data_module,
        "load_smf_from_h5",
        lambda chromosomes: (
            ["valid", "invalid"],
            [
                torch.tensor([[1]], dtype=torch.int8),
                torch.tensor([[0]], dtype=torch.int8),
            ],
            [torch.tensor([2]), torch.tensor([1])],
            [source, source],
        ),
    )
    monkeypatch.setattr(
        "smf_net.data.load_dna",
        lambda fasta_path, regions, region_length: torch.stack(
            [dna_str_to_tensor("AGCAAAAA"), dna_str_to_tensor("ACGAAAAA")]
        ),
    )

    data_module.setup("validate")

    dataset = data_module.val_dataset
    assert dataset.regions == ["valid"]
    assert torch.equal(dataset.smf[0], torch.tensor([[1]], dtype=torch.int8))
    assert torch.equal(dataset.pos_list[0], torch.tensor([2]))
    assert torch.equal(dataset.dna[0], dna_str_to_tensor("AGCAAAAA"))
    assert dataset.sources == [source]
    assert dataset.assay_type == "gch_accessibility"
    assert dataset.rc_augmentation is True


def test_data_module_only_sets_up_requested_split(tmp_path, monkeypatch):
    h5_path = tmp_path / "regions.h5"
    split_regions = {
        "train": "chr1_1-12",
        "val": "chr2_1-12",
        "test": "chr3_1-12",
    }
    with h5py.File(h5_path, "w") as h5_file:
        pos_group = h5_file.create_group("smf_pos")
        mat_group = h5_file.create_group("smf_mat")
        for region in split_regions.values():
            pos_group.create_dataset(region, data=[0, 2, 4, 6, 8, 10])
            mat_group.create_dataset(
                region,
                data=torch.randint(0, 2, (4, 6), dtype=torch.int8).numpy(),
            )

    dna_loads = []

    def fake_load_dna(fasta_path, regions, region_length):
        dna_loads.append(list(regions))
        return dna_str_to_tensor("CG" * 6).repeat(len(regions), 1)

    monkeypatch.setattr("smf_net.data.load_dna", fake_load_dna)
    cfg = SparseDataModuleConfig(
        train_chrm=["chr1"],
        val_chrm=["chr2"],
        test_chrm=["chr3"],
        fasta_path="unused.fa",
        h5_path=str(h5_path),
        batch_size=2,
        num_workers=0,
        h5_region_length=12,
        model_input_length=8,
    )
    data_module = SparseSMFDataModule(cfg)

    data_module.setup("validate")

    assert data_module.train_dataset is None
    assert data_module.test_dataset is None
    assert data_module.val_dataset.train is False
    assert data_module.val_dataset.sources[0].kind == "observed_h5"
    assert data_module.val_dataset.sources[0].path == str(h5_path.resolve())
    assert data_module.val_dataset.get_region(0).source.path == str(h5_path.resolve())
    assert data_module.val_dataset.regions == [split_regions["val"]]
    assert dna_loads == [[split_regions["val"]]]

    val_dataset = data_module.val_dataset
    data_module.setup("validate")
    assert data_module.val_dataset is val_dataset
    assert dna_loads == [[split_regions["val"]]]

    data_module.setup("test")
    assert data_module.train_dataset is None
    assert data_module.test_dataset.regions == [split_regions["test"]]
    assert dna_loads[-1] == [split_regions["test"]]


def test_data_module_setup_none_loads_all_splits(tmp_path, monkeypatch):
    h5_path = tmp_path / "regions.h5"
    with h5py.File(h5_path, "w") as h5_file:
        pos_group = h5_file.create_group("smf_pos")
        mat_group = h5_file.create_group("smf_mat")
        for chrom in ("chr1", "chr2", "chr3"):
            region = f"{chrom}_1-12"
            pos_group.create_dataset(region, data=[0, 2])
            mat_group.create_dataset(region, data=[[0, 1]])

    monkeypatch.setattr(
        "smf_net.data.load_dna",
        lambda fasta_path, regions, region_length: dna_str_to_tensor("CG" * 6).repeat(
            len(regions), 1
        ),
    )
    cfg = SparseDataModuleConfig(
        train_chrm=["chr1"],
        val_chrm=["chr2"],
        test_chrm=["chr3"],
        fasta_path="unused.fa",
        h5_path=str(h5_path),
        batch_size=1,
        num_workers=0,
        h5_region_length=12,
        model_input_length=8,
    )
    data_module = SparseSMFDataModule(cfg)

    data_module.setup(None)

    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert data_module.test_dataset is not None
    assert data_module.train_dataset.sources is None
    assert data_module.val_dataset.sources is not None
    assert data_module.test_dataset.sources is not None


def test_data_module_thins_train_and_val_but_not_test(tmp_path, monkeypatch):
    h5_path = tmp_path / "regions.h5"
    with h5py.File(h5_path, "w") as h5_file:
        pos_group = h5_file.create_group("smf_pos")
        mat_group = h5_file.create_group("smf_mat")
        for chrom in ("chr1", "chr2", "chr3"):
            region = f"{chrom}_1-12"
            pos_group.create_dataset(region, data=[0, 2])
            mat_group.create_dataset(
                region,
                data=torch.arange(50, dtype=torch.int8).unsqueeze(1).repeat(1, 2),
            )

    monkeypatch.setattr(
        "smf_net.data.load_dna",
        lambda fasta_path, regions, region_length: dna_str_to_tensor("CG" * 6).repeat(
            len(regions), 1
        ),
    )
    cfg = SparseDataModuleConfig(
        train_chrm=["chr1"],
        val_chrm=["chr2"],
        test_chrm=["chr3"],
        fasta_path="unused.fa",
        h5_path=str(h5_path),
        num_workers=0,
        h5_region_length=12,
        model_input_length=8,
        train_val_read_fraction=0.5,
        read_downsample_seed=7,
    )

    data_module = SparseSMFDataModule(cfg)
    data_module.setup(None)
    validate_only_data_module = SparseSMFDataModule(cfg)
    validate_only_data_module.setup("validate")

    assert 0 < data_module.train_dataset.num_reads[0] < 50
    assert 0 < data_module.val_dataset.num_reads[0] < 50
    assert data_module.test_dataset.num_reads[0] == 50
    assert torch.equal(data_module.val_dataset.smf[0], validate_only_data_module.val_dataset.smf[0])


@pytest.mark.parametrize(("limit", "expected_regions"), [(1, 1), (10, 2)])
def test_region_limit_is_a_maximum(tmp_path, monkeypatch, limit, expected_regions):
    h5_path = tmp_path / "regions.h5"
    with h5py.File(h5_path, "w") as h5_file:
        pos_group = h5_file.create_group("smf_pos")
        mat_group = h5_file.create_group("smf_mat")
        for start in (1, 13):
            region = f"chr2_{start}-{start + 11}"
            pos_group.create_dataset(region, data=[0, 2])
            mat_group.create_dataset(region, data=[[0, 1]])

    monkeypatch.setattr(
        "smf_net.data.load_dna",
        lambda fasta_path, regions, region_length: dna_str_to_tensor("CG" * 6).repeat(
            len(regions), 1
        ),
    )
    cfg = SparseDataModuleConfig(
        train_chrm=["chr1"],
        val_chrm=["chr2"],
        test_chrm=["chr3"],
        fasta_path="unused.fa",
        h5_path=str(h5_path),
        batch_size=1,
        num_workers=0,
        max_regions_per_h5_per_split=limit,
        h5_region_length=12,
        model_input_length=8,
    )
    data_module = SparseSMFDataModule(cfg)

    data_module.setup("validate")

    assert len(data_module.val_dataset.regions) == expected_regions


def test_uniform_sampling_replacement_when_num_reads_small():
    smf_pos = torch.arange(4, dtype=torch.long)
    smf_matrix = torch.randint(0, 2, (2, 4), dtype=torch.int8)
    dna_tensor = dna_str_to_tensor("CGCG").unsqueeze(0)
    dataset = SparseSMFDataset(
        smf_list=[smf_matrix],
        pos_list=[smf_pos],
        dna=dna_tensor,
        reads_per_region=5,
        train=False,
        model_input_length=4,
    )

    generator = torch.Generator().manual_seed(0)
    read_idx = dataset._sample_uniform_read_idx(num_reads=2, generator=generator)
    assert read_idx.shape == (5,)


def test_single_threaded_torch_restores_the_thread_count_even_on_error():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with single_threaded_torch():
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 2

        with pytest.raises(RuntimeError), single_threaded_torch():
            raise RuntimeError
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_load_dna_reads_exclusive_and_inclusive_end_names_alike(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nACGTACGTAC\n")
    # Bases 3..6 (1-based), named with an exclusive end and with an inclusive end.
    dna = load_dna(str(fasta), ["chr1:3-7", "chr1:3-6"], region_length=4)
    expected = dna_str_to_tensor("GTAC")
    assert torch.equal(dna, torch.stack([expected, expected]))
