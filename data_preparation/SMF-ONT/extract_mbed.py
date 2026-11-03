#!/usr/bin/env python3
"""Extract SMFNet reads-by-positions H5 from indexed five-column mBED.

Input columns: chromosome, read_start (0-based), read_end, read_id, delta string
(e.g. 10m5u4x). Each delta is relative to the preceding event:
event coordinate (1-based) = read_start + 1 + cumulative offsets.
Requires numpy, h5py, pysam and an indexed FASTA and bgzip/tabix-indexed mBED.

python extract_mbed.py --bed regions.bed --mbed calls.bed.gz --fasta ref.fa --out reads.h5
Use --bed-coordinates one-based for 1-based, closed region coordinates.
Default BED coordinates are 0-based, half-open; positions are 0-based relative
to the first base and keys are chr:<first base 1-based>-<last base 1-based>.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing as mp
import os
from pathlib import Path
import re
import tempfile
import time

import h5py
import numpy as np
import pysam

DELTA_TOKEN_PATTERN = re.compile(r"(\d+)([mux])")
CALL_ENCODING = {"m": 0, "u": 1, "x": -1}
REFERENCE_BLOCK_SIZE = 262144


def initialize(mbed, fasta):
    """Open process-local readers; file handles are not shared across workers."""
    global MBED_INDEX, REFERENCE_FASTA
    MBED_INDEX = pysam.TabixFile(mbed)
    REFERENCE_FASTA = pysam.FastaFile(fasta)
    reference_block.cache_clear()
    decode.cache_clear()


@lru_cache(maxsize=64)
def reference_block(chrom, block):
    return REFERENCE_FASTA.fetch(
        chrom,
        block * REFERENCE_BLOCK_SIZE,
        min(
            (block + 1) * REFERENCE_BLOCK_SIZE,
            REFERENCE_FASTA.get_reference_length(chrom),
        ),
    ).upper()


def base(chrom, position):
    """Reference base at a 1-based coordinate, cached in bounded genomic blocks."""
    if position < 1 or position > REFERENCE_FASTA.get_reference_length(chrom):
        return "N"
    block, offset = divmod(position - 1, REFERENCE_BLOCK_SIZE)
    return reference_block(chrom, block)[offset]


@lru_cache(maxsize=128)
def decode(chrom, start, signal):
    """Decode delta offsets into absolute coordinates, call values and strand."""
    tokens = DELTA_TOKEN_PATTERN.findall(signal)
    if not tokens or sum(len(delta) + 1 for delta, _ in tokens) != len(signal):
        raise ValueError(f"Malformed mBED delta string at {chrom}:{start}")
    positions = (
        start
        + 1
        + np.cumsum(np.fromiter((int(delta) for delta, _ in tokens), dtype=np.int64))
    )
    values = np.fromiter((CALL_ENCODING[symbol] for _, symbol in tokens), dtype=np.int8)
    # Infer the strand from the sequence context of the first five events.
    # An ambiguous strand cannot be mapped reliably to cytosine anchors.
    plus = minus = True
    for position in positions[:5]:
        left, center, right = (
            base(chrom, int(position) - 1),
            base(chrom, int(position)),
            base(chrom, int(position) + 1),
        )
        plus &= center == "C" and (left == "G" or right == "G")
        minus &= center == "G" and (left == "C" or right == "C")
    return positions, values, "+" if plus else "-" if minus else "?"


def extract_region(region):
    """Collect overlapping molecules and align calls to reference cytosines."""
    chrom, start, end = region  # 1-based inclusive
    key = f"{chrom}:{start}-{end}"
    sequence_start = max(1, start - 2)
    sequence = REFERENCE_FASTA.fetch(
        chrom,
        sequence_start - 1,
        min(end + 2, REFERENCE_FASTA.get_reference_length(chrom)),
    ).upper()

    def local(position):
        index = position - sequence_start
        return sequence[index] if 0 <= index < len(sequence) else "N"

    all_positions, read_ids, rows = set(), [], {}
    skipped = 0
    for record in MBED_INDEX.fetch(chrom, start - 1, end):
        fields = record.split("\t")
        if len(fields) < 5:
            raise ValueError("mBED requires at least five columns")
        read_id = fields[3]
        positions, values, strand = decode(chrom, int(fields[1]), fields[4])
        if strand == "?":
            skipped += 1
            continue
        first, last = np.searchsorted(positions, [start, end], side="left")
        last = np.searchsorted(positions, end, side="right")
        # Repeated coordinates keep the last call, retaining insertion order.
        events = dict(zip(positions[first:last].tolist(), values[first:last].tolist()))
        # Minus-strand events are shifted onto the corresponding cytosine.
        anchors, row = {}, {}
        had_event = False
        for position, value in events.items():
            left, center, right = (
                local(position - 1),
                local(position),
                local(position + 1),
            )
            targets = []
            if strand == "+":
                if center == "C" and (left == "G" or right == "G"):
                    targets = [position]
            elif center == "G":
                if left == "C" and right == "C":
                    targets = [position + 1, position - 1]
                elif right == "C":
                    targets = [position + 1]
                elif left == "C" and position - 2 not in anchors:
                    targets = [position - 1]
            if targets:
                anchors[position] = targets
                had_event = True
            for anchor in targets:
                if start <= anchor <= end:
                    row[anchor - start] = value
                    all_positions.add(anchor - start)
        if had_event:
            read_ids.append(read_id)
            rows[read_id] = (
                row  # For repeated read IDs, the final record supplies the calls.
            )
    # Use one shared, sorted position axis; absent calls remain -1.
    relative_positions = np.array(sorted(all_positions), dtype=np.int64)
    matrix = np.full((len(read_ids), len(relative_positions)), -1, dtype=np.int8)
    for read_index, read_id in enumerate(read_ids):
        row = rows[read_id]
        if row:
            columns = np.searchsorted(relative_positions, list(row))
            matrix[read_index, columns] = list(row.values())
    return key, relative_positions, matrix, read_ids, skipped


def bed_regions(path, coordinates):
    """Read unique regions and normalize them to 1-based inclusive coordinates."""
    seen = set()
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"BED line {line_number}: expected at least three columns"
                )
            start, end = int(fields[1]), int(fields[2])
            if coordinates == "bed":
                start += 1
            if start < 1 or end < start:
                raise ValueError(f"Invalid region at BED line {line_number}")
            region = (fields[0], start, end)
            if region not in seen:
                seen.add(region)
                yield region


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bed", required=True)
    parser.add_argument("--mbed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--bed-coordinates", choices=["bed", "one-based"], default="bed"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compression", type=int, choices=range(10), default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    target = Path(args.out)
    if target.resolve() in {
        Path(p).resolve() for p in [args.bed, args.mbed, args.fasta]
    }:
        parser.error("output must differ from input files")
    if target.exists() and not args.overwrite:
        parser.error("output exists; use --overwrite")
    if not Path(args.fasta + ".fai").exists():
        parser.error("FASTA needs an existing .fai index (samtools faidx ref.fa)")
    regions = list(bed_regions(args.bed, args.bed_coordinates))
    if not regions:
        parser.error("BED contains no regions")
    # Validate all contigs/bounds before creating output or spawning workers.
    with pysam.FastaFile(args.fasta) as reference_handle, pysam.TabixFile(
        args.mbed
    ) as mbed_handle:
        for chrom, start, end in regions:
            if (
                chrom not in reference_handle.references
                or end > reference_handle.get_reference_length(chrom)
            ):
                parser.error(f"Invalid FASTA interval: {chrom}:{start}-{end}")
        missing = set(r[0] for r in regions) - set(mbed_handle.contigs)
        if missing:
            parser.error(f"Contigs absent from mBED index: {sorted(missing)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish the HDF5 file only after every region has been written.
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".partial", dir=target.parent
    )
    os.close(file_descriptor)
    started = time.perf_counter()
    executor = None
    try:
        if args.workers > 1:
            executor = ProcessPoolExecutor(
                args.workers,
                mp_context=mp.get_context("spawn"),
                initializer=initialize,
                initargs=(args.mbed, args.fasta),
            )
        else:
            initialize(args.mbed, args.fasta)
        with h5py.File(temporary, "w") as h5:
            groups = {
                name: h5.create_group(name)
                for name in ["smf_mat", "smf_pos", "read_id"]
            }
            h5.attrs.update(
                matrix_order="reads_by_positions",
                position_origin="first_base_zero",
                region_coordinates="one_based_closed",
                encoding="-1=missing,0=methylated,1=unmethylated",
                input_bed_coordinates=args.bed_coordinates,
            )
            # Submit only a bounded window; completed matrices cannot pile up.
            window = 2 * args.workers
            for begin in range(0, len(regions), window):
                batch = regions[begin : begin + window]
                results = (
                    executor.map(extract_region, batch)
                    if executor
                    else map(extract_region, batch)
                )
                for key, positions, matrix, read_ids, skipped in results:
                    kwargs = (
                        dict(
                            compression="gzip",
                            compression_opts=args.compression,
                            shuffle=True,
                        )
                        if matrix.size and args.compression
                        else {}
                    )
                    groups["smf_mat"].create_dataset(key, data=matrix, **kwargs)
                    groups["smf_pos"].create_dataset(key, data=positions)
                    groups["read_id"].create_dataset(
                        key,
                        data=np.array(read_ids, dtype=object),
                        dtype=h5py.string_dtype(),
                    )
                    print(
                        f"{key}: {matrix.shape[0]} reads, {positions.size} sites; {skipped} unaligned records",
                        flush=True,
                    )
        os.replace(temporary, target)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        else:
            MBED_INDEX.close()
            REFERENCE_FASTA.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {target} in {time.perf_counter()-started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
