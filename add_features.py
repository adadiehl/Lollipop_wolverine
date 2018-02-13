#!/usr/bin/python
# AGD: Corrected shebang line -- now runs without python prefix on most systems!

# Author: Yan Kai
# This script is to prepare features for the generated loops for training purpose.
# Two inputs: 1. The generated positive and negative loops in the format: chrom+start1+start2+length+response
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

""" Below were added by AGD """
import tabix as tb
import multiprocessing
import ctypes
from lib import add_motif_pattern, add_peak_feature, add_bigWig_feature, add_gene_expression

"""
Function defs
"""

def add_local_feature(signal, train, BED):
    """
    This function is to calculate the signal values on summit positions (summit +/- 2kb)
    """
    sys.stderr.write("\t\tPreparing the local features of {} ...".format(str(signal))) # AGD: Use stderr and str.format
    extension = opt.extension
    fragment = 150 # This is the size of the ChIP fragment, usually it is 150.
    shift = fragment/2
    BED_reader = open(BED,'r')
    read_info = {}  # read_info = {'chrom':[start1, start2,...]}  actually 'start' here is the mid-point of a fragment

    read_number = 0
    for line in BED_reader:
        read_number += 1
        pline = line.strip()
        sline = pline.split('\t')
        chrom = sline[0]
        start = int(sline[1])
        end = int(sline[2])
        strand = sline[5]

        if chrom not in read_info.keys():
            read_info[chrom] = []
        if (strand == '+'):
            start = start + shift
        else:
            start = end - shift

        read_info[chrom].append(start)
    BED_reader.close()

    # make sure reads are sorted
    for chrom in read_info.keys():
        read_info[chrom] = sorted(read_info[chrom])
        
    RPKMs1 = []
    RPKMs2 = []
    for index, row in train.iterrows():
        chrom = row['chrom']
        if opt.extension > 0:
            start1 = row['peak1'] - opt.extension
            start2 = row['peak2'] - opt.extension
            end1 = row['peak1'] + opt.extension
            end2 = row['peak2'] + opt.extension
        else:
            start1 = row['start1']
            start2 = row['start2']
            end1 = row['end1']
            end2 = row['end2']
        len1 = end1 - start1
        len2 = end2 - start2

        count1 = bisect.bisect_right(read_info[chrom], end1) - bisect.bisect_left(read_info[chrom], start1)
        count2 = bisect.bisect_right(read_info[chrom], end2) - bisect.bisect_left(read_info[chrom], start2)

        RPKM1 = float(count1)/(float(read_number)*len1)*1000000000
        RPKM2 = float(count2)/(float(read_number)*len2)*1000000000

        RPKMs1.append((RPKM1+RPKM2)/2.0)
        RPKMs2.append(np.std([RPKM1, RPKM2]))

    signal1 = 'avg_'+str(signal)
    signal2 = 'std_'+str(signal)
    train[signal1] = pd.Series(RPKMs1, index = train.index)
    train[signal2] = pd.Series(RPKMs2, index = train.index)
    return train


def add_regional_feature_by_reads(signal, train, anchors, BED):
    """
    This function is to add the in-between and loop-flanking features for an interacting pair from raw reads BED files, whose format is:
    chrom+start+end.

    """
    print "\t\tPreparing the in-between and loop-flanking features of "+str(signal)+'...'
    BED = open(BED, 'r')
    shift = 75
    signal_dic = {}  # signal_dic = {"chrXX":[start1, start2, ...]} 'start' here are mid-point of one fragment
    read_number = 0
    for line in BED:
        read_number += 1
        pline = line.strip()
        sline = pline.split('\t')
        chrom = sline[0]
        start = int(sline[1])
        end = int(sline[2])
        strand = sline[5]

        if chrom not in signal_dic.keys():
            signal_dic[chrom] = []
        if strand == '+':
            start = start + shift
        else:
            start = end - shift
        signal_dic[chrom].append(start)
    BED.close()
    
    # make sure reads are sorted
    for chrom in signal_dic.keys():
        signal_dic[chrom] = sorted(signal_dic[chrom])

    in_between = []
    upstream = []
    downstream = []
    for index, row in train.iterrows():
        chrom = row['chrom']
        start1 = row['peak1']
        start2 = row['peak2']

        index1 = anchors[chrom].index(start1)
        index2 = anchors[chrom].index(start2)
        if index1 != 0:
            up_motif = anchors[chrom][index1 - 1]
            up_count = bisect.bisect_right(signal_dic[chrom], start1) - bisect.bisect_left(signal_dic[chrom], up_motif)
            up_strength = float(up_count)/float(abs(up_motif-start1)*read_number)*1e+9
        else:
            up_strength = 0
        upstream.append(up_strength)
        if index2 != (len(anchors[chrom])-1):
            down_motif = anchors[chrom][index2 + 1]
            down_count = bisect.bisect_right(signal_dic[chrom], down_motif) - bisect.bisect_left(signal_dic[chrom], start2)
            down_strength = float(down_count)/float(abs(down_motif-start2)*read_number)*1e+9
        else:
            down_strength = 0
        downstream.append(down_strength)

        strength = 0
        count = bisect.bisect_right(signal_dic[chrom], start2) - bisect.bisect_left(signal_dic[chrom], start1)

        strength = float(count)/float(abs(start2-start1)*read_number)*1e+9
        in_between.append(strength)

    in_between_signal = signal+'_in-between'
    train[in_between_signal] = pd.Series(in_between, index=train.index)

    upstream_signal = signal+'_left'
    train[upstream_signal] = pd.Series(upstream, index=train.index)

    downstream_signal = signal+'_right'
    train[downstream_signal] = pd.Series(downstream, index=train.index)

    return train




def main(argv):
	parser = OptionParser()
	parser.add_option("-i", "--training", action="store", type="string", dest="training", metavar="<file>", help="the training interactions generated in previous step")
	parser.add_option("-t", "--table", action="store", type="string", dest="info_table", metavar="<file>", help="the infomation table contains the paths for necessary files.")
	parser.add_option("-o", "--outfile", action="store", type="string", dest="outfile", metavar="<file>", help="the output file with all the interactions and calculated features.")
        parser.add_option('-p', '--procs', type=int, default=1,
                          help='Number of processors to use. Default 1.')
        parser.add_option('-e', '--extension', type=int, default=1000,
                          help='Size of the extesion window to append up and downstream of each anchor peak for signal calculation. Default 1000 (for window size 2000). Set to 0 to use actual feature boundaries.')
        parser.add_option('-m', '--motif_extension', type=int, default=500,
                          help='Size of extension window to append when finding motif features. Default 500.')
        parser.add_option('-z', '--cons_extension', type=int, default=20,
                          help='Size of extension window to append when calculating conservation. Default 20.')
        parser.add_option('-a', '--collapse_peaks', type='choice', choices=["max", "avg", "min", "sum"], default='max',
                          help='How to handle multiple overlapping peak features. Allowed values: max (use maximum score), avg (average all scores), min (use lowest score), sum (use sum of all scores). Default = max.')        
        
        
	(opt, args) = parser.parse_args(argv)
	if len(argv) < 6:
		parser.print_help()
        	sys.exit(1)

	start_time = time.time()
	chroms = GenomeData.hg19_chroms
	#chroms = ['chr1']
	train = pd.read_table(opt.training)
	train = train.sort_values(by=['chrom1','peak1','peak2'], axis = 0, ascending=[1,1,1])
	info_table = pd.read_table(opt.info_table)
	anchors = {} # anchors = {'chr':set(summit1, summit2,)}
	
        # Generate the anchors pool from the anchors of positive loops
        sys.stderr.write("Preparing the anchor pool...\n")
	for index,row in train.iterrows():
	        chrom = row['chrom1']
	        if chrom not in anchors.keys():
	            anchors[chrom] = set()
	        anchors[chrom].add(row['peak1'])
	        anchors[chrom].add(row['peak2'])
        sys.stderr.write("Sorting the anchors...\n")
	for chrom in anchors.keys():
		anchors[chrom] = list(anchors[chrom])
		anchors[chrom].sort()

        sys.stderr.write("Processing features...\n")
	for index, row in info_table.iterrows():
		signal = row['Signal']
		Format = row['Format']
		Path = row['Path']

		if signal == 'Motif':  # AGD: Don't use parens around conditionals
                    sys.stderr.write("\tAdding motif annotations...\n")
		    train = add_motif_pattern(train, Path, opt)

                # AGD: Removed redundant loop through info_table
		elif signal == 'PhastCon':
                    sys.stderr.write("\tAdding PhastCons conservation...\n")
		    train = add_bigWig_feature(train, Path, opt)
		elif signal == 'Gene expression':
                    sys.stderr.write("\tAdding gene expression...\n")
		    train = add_gene_expression(train, Path)
		else:
                    sys.stderr.write("\tProcessing {} features\n".format(signal))
		    if Format == 'bed':
			train = add_local_feature(signal, train, Path)
			train = add_regional_feature_by_reads(signal, train, anchors, Path)
                    elif Format == 'narrowPeak':
                        train = add_peak_feature(signal, train, Path, opt)


	train.to_csv(opt.outfile,sep='\t', index=False)
	end_time = time.time()
	elapsed = end_time-start_time
	print "Time elapased: "+str(elapsed)+' seconds'


if __name__ == "__main__":
	main(sys.argv)
