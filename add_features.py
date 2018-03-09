#!/usr/bin/python
# AGD: Corrected shebang line -- now runs without python prefix on most systems!

# Author: Yan Kai
# This script is to prepare features for the generated loops for training purpose.
# Two inputs: 1. The generated positive and negative loops in the format: chrom+peak1+peak2+length+response
#             2. The infomation table that contains the complete path for necessary raw BED file and peak file.
# Output is: The training data, i.e the interactions plus all the listed features.

# Optimized/bastardized by Adam Diehl (AGD), adadiehl@umich.edu, 1/24/2018.
# Changes from Yan Kai original noted.

import sys, re, os
import numpy as np
import GenomeData
import collections
import time
from optparse import OptionParser
import pandas as pd
import HTSeq
import bisect

import tabix as tb # Added by AGD
import multiprocessing
import ctypes

import lib

"""
Function defs
"""

def main(argv):
	parser = OptionParser()
	parser.add_option("-i", "--training", action="store", type="string", dest="training", metavar="<file>", help="the training interactions generated in previous step")
	parser.add_option("-t", "--table", action="store", type="string", dest="info_table", metavar="<file>", help="the infomation table contains the paths for necessary files.")
	parser.add_option("-o", "--outfile", action="store", type="string", dest="outfile", metavar="<file>", help="the output file with all the interactions and calculated features.")
        parser.add_option('-p', '--procs', type=int, default=1,
                          help='Number of processors to use. Default 1.')
        parser.add_option('-e', '--extension', type=int, default=2000,
                          help='Size of the extesion window to append up and downstream of each anchor peak for signal calculation. Default 2000. Set to 0 to use actual element boundaries.')
        parser.add_option('-m', '--motif_extension', type=int, default=500,
                          help='Size of extension window to append when finding motif features. Default 500. Set to 0 to use actual element boundaries.')
        parser.add_option('-z', '--cons_extension', type=int, default=20,
                          help='Size of extension window to append when calculating conservation. Default 20. Set to 0 to use actual element boundaries.')
        parser.add_option('-c', '--collapse_peaks', type='choice', choices=["max", "avg", "min", "sum"], default='max',
                          help='How to handle multiple overlapping peak features. Allowed values: max (use maximum score), avg (average all scores), min (use lowest score), sum (use sum of all scores). Default = max.')        
        parser.add_option('-n', '--no_in_between_peaks', dest="in_between", action='store_false', default=True,
                          help='Do not include "in-between" peaks in training and predictions.')
        parser.add_option('-g', '--no_flanking_peaks', dest="flanking", action='store_false', default=True,
                          help='Do not include "upstream" and "downstream" peaks in training and predictions.')
        
	(opt, args) = parser.parse_args(argv)
	if len(argv) < 6:
		parser.print_help()
        	sys.exit(1)

	start_time = time.time()
	chroms = GenomeData.hg19_chroms
	train = pd.read_table(opt.training)
	train = train.sort_values(by=['chrom','peak1','peak2'], axis = 0, ascending=[1,1,1])

        # Read in the signals table
        signal_table, signals = lib.load_signals_table(opt.info_table)
        
        # Generate the anchors pool from the anchors of positive loops
        sys.stderr.write("Preparing the anchor pool...\n")
        anchors = {} # anchors = {'chr':set(summit1, summit2,)}
	for index,row in train.iterrows():
	    chrom = row['chrom']
	    if chrom not in anchors.keys():
	        anchors[chrom] = set()
	    anchors[chrom].add(row['peak1'])
	    anchors[chrom].add(row['peak2'])
        sys.stderr.write("Sorting the anchors...\n")
	for chrom in anchors.keys():
	    anchors[chrom] = list(anchors[chrom])
	    anchors[chrom].sort()

        # Prepare reads info for read-based features (allows use of shared lib to calculate features)
        sys.stderr.write('Preparing reads information...\n')
        read_info, read_numbers = lib.prepare_reads_info(signal_table)
        
        sys.stderr.write("Processing features...\n")
        train = lib.prepare_features_for_interactions(train, anchors, signal_table, read_info, read_numbers, opt)
	train.to_csv(opt.outfile,sep='\t', index=False)
	end_time = time.time()
	elapsed = end_time-start_time
	print "Time elapased: "+str(elapsed)+' seconds'


if __name__ == "__main__":
	main(sys.argv)
