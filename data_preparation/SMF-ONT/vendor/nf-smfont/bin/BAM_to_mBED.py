#! /usr/bin/env python3
import os
import sys
import argparse
import itertools
from pathlib import Path
from modbampy import ModBam
import numpy as np
import gzip
import multiprocessing as mp

from SMFONT_utils import Converter, AccessibilityEntry, p_to_llr, chunk_input_genome
from importlib.metadata import version

"""
Need to be clean up from old REne SMF accessibility code
"""

def parseArgs():
	parser = argparse.ArgumentParser(description='Convert BAM files into methylBED Krebs lab (Timp lab std...)')
	parser.add_argument('-v', '--verbose', action="store_true", required=False, default = False, help="verbose output")
	parser.add_argument('-i', '--input', type=str, required=True, help="input BAM file")
	parser.add_argument('-r', '--reference', type=str, required=True, help="Fasta file of the reference genome")
	parser.add_argument('-m', '--motifs',type=str, required=True, action='append', nargs='+', help="DNA Motif of expected methylation site")
	parser.add_argument('-o', '--out', type=str, required=False, help="output BED file")
	parser.add_argument('-w', '--overwrite', action="store_true", required=False, default = False, help="Overwrite output file")
	parser.add_argument('-t', '--threads', type=int, required=False, default=2, help="worker threads")
	parser.add_argument('-c', '--chunk_size', type=float, required=False, default=5e6, help="size (bp) of genome tiles")
	parser.add_argument('-l', '--llr', type=float, required=False, default=2.0, help="Likelihood ratio used to convert the Remora probability (default : 2)")
	args = parser.parse_args()
	return args


def get_version():
    modulenames = set(sys.modules) & set(globals())
    with open('python_packages.txt', 'a') as f:
        for m in modulenames:
            try:
                f.write(f"    python-{m}: {globals()[m].__version__}\n")
            except AttributeError:
                try:
                    f.write(f"    python-{m}: {version(m)}\n")
                except ModuleNotFoundError:
                    f.write(f"    python-{m}: ????\n") 


def extractMeth(mq, bam, chrom, start, end, reference, llr, motifs, verbose):
	if verbose:
		print("Meth Reader Started.")

	with Converter(reference, exclude_zero_calls=True, llr_threshold=llr, motifs=motifs) as converter:
		for read in ModBam(bam).reads(chrom, start, end):
			llrs = np.zeros((read.reference_end - read.reference_start))
			for pos_mod in read.mod_sites:
				rpos = pos_mod.rpos
				if rpos < 0:
					continue
				p = float(pos_mod.qual) / 256
				llrs[rpos - read.reference_start] = p_to_llr(np.clip(p, 0.000001, 0.999999))
			read_entry = AccessibilityEntry(chrom,read.reference_start,read.reference_end,
				not read.is_reverse,read.query_name,llrs)
			timp_line = converter.format_line(converter.convert_line(read_entry))
			if timp_line is None:
				continue
			mq.put(f"{timp_line}\n")


def writer(mq, out, overwrite, verbose):
	if verbose:
		print("Writer Started.")

	if out:
		ext = out.strip().split(".")[-1]
		if ext in ["gz", "bgz"] :
			output = Path(out)
		else:
			output = Path(out).joinpath(".gz")
		if os.path.exists(output):
			if overwrite:
				if verbose:
					print("Removing exciting output file.", file=sys.stderr)
				os.remove(output) 
			else:
				sys.exit("The output file {} is exciting!\nPlease use the option -w to overwrite it.".format(output))
		out_f = gzip.open(output, "at")
	else:
		out_f = sys.stdout

	while True:
		m = mq.get()
		if m == '#done#':
			#Notify coordinator process of task completion      
			#coord_q.put('WRITE_DONE')
			break
		out_f.write(m)
		out_f.flush()


def main():
	args=parseArgs()

	if args.threads < 2:
		sys.exit("At least 2 threads required!")
	
	# init multiprocessing
	manager = mp.Manager()
	mq = manager.Queue()

	bam = args.input
	bai = bam + ".bai"

	motifs = list(itertools.chain(*args.motifs))
	ref_chunk = chunk_input_genome(args.reference, args.chunk_size)

	# init writer
	wr_P = mp.Pool(1)
	wr_P.apply_async(writer, args=(mq, args.out, args.overwrite, args.verbose))

	jobs = []
	p = mp.Pool(args.threads - 1)
	
	for coord in ref_chunk:
		r = p.apply_async(extractMeth, args=(mq, bam, coord[0], coord[1], coord[2], args.reference, args.llr, motifs, args.verbose))
		jobs.append(r)
	
	for job in jobs:
		job.get()

	mq.put('#done#') #
	p.close()
	p.join()
	get_version()


if __name__=="__main__":
	main()
