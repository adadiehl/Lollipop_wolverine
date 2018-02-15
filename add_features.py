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

from lib import prepare_anchors

"""
Globals (Needed for multiprocessing)
"""
pattern = []
scores1 = []
scores2 = []
lock = multiprocessing.Lock()

"""
Function defs
"""
def _init_peaks(s1, s2):
    """
    Multiprocessing init function for narrowPeaks.
    """
    global scores1
    global scores2
    scores1 = s1
    scores2 = s2

def _init_motifs(s1, s2, p):
    """
    Multiprocessing init function for motifs.
    """
    global scores1
    global scores2
    global pattern
    scores1 = s1
    scores2 = s2
    pattern = p
    

def assign_motif_pattern(strand1, strand2):
    if (strand1 == '-' and strand2 == '+'):
        return '2'
    elif (strand1 == '+' and strand2 == '-'):
        return '4'
    else:
        return '3'


def get_features(chrom, start, end, feats_f, names, dtypes):
    """                                                                         
    Get a pandas dataframe of features within a given anchor. Uses Tabix.       
    Added by AGD, 1/25/2018.

    Input:                                                                      
        chrom = chromosome                                                      
        start = chromStart                                                      
        end = chromEnd                                                          
        feats_f = Tabix file handle for feature annotation file.                
        names = list of column names for output tables.                         
        dtypes = list of data types for each column.                            
    Output:                                                                     
        A pandas dataframe for the given genomic interval, with column          
        names and types set accordingly.                                        
    """
    if len(names) != len(dtypes):
        sys.stderr.write("get_features: ERROR -- names and dtypes must have same length!\n")
        exit(1)
    feats = feats_f.querys( '{}:{}-{}'.format(chrom,
                                              start,
                                              end) )
    # Convert to a pandas dataframe with the supplied column names
    feats = pd.DataFrame(list(feats), columns = names)
    # Convert datatypes for any numeric columns (strings should be fine as is)
    for i in range(0, len(names)):
        d = dtypes[i]
        if d == 'int' or d == 'int64' or d == 'float' or d == 'float64':
            feats[names[i]] = pd.to_numeric(feats[names[i]])
    return feats


def find_motif_pattern(map_args, def_param=(scores1, scores2, pattern)):
    # AGD: motif now holds Tabix handle for motif annotations file
    """
     Input:
         anchor = HTSeq.GenomicInterval(chrom,summit-ext, summit+ext,'.')
         motif = {'chromXX':{start:(strand, score)}}
     Output:
         a tuple (pattern, avg_motif_strength, std_motif_strength)
     Rules to assign motif pattern:
     1. Both anchors have no motif, assign 0;
     2. One anchor has no motif, no matter how many motifs the other anchor may have, assign 1;
     3. Both anchors have 1 motif: no ambuguity, divergent=2;tandem=3; convergent=4
     4. Anchors have multiple motifs: in each anchor, choose the one with the highest motif strength
    """
    (i, train, Peak, opt) = map_args
    row = train.iloc[i]
    anchor1, anchor2 = prepare_anchors(row, opt.motif_extension)
    motif = tb.open(Peak)

    # AGD: Get motifs in range with call to Tabix-indexed motif file.
    # feats1 and feats2 are pandas DataFrame objects with given column
    # names and data types. No more inefficient loops!
    feats1 = get_features(anchor1.chrom,
                          anchor1.start,
                          anchor1.end,
                          motif,
                          ["chrom",
                           "start",
                           "end",
                           "name",
                           "score",
                           "strand"],
                          ["string",
                           "int64",
                           "int64",
                           "string",
                           "float64",
                           "string"])
    feats2 = get_features(anchor2.chrom,
                          anchor2.start,
                          anchor2.end,
                          motif,
                          ["chrom",
                           "start",
                           "end",
                           "name",
                           "score",
                           "strand"],
                          ["string",
                           "int64",
                           "int64",
                           "string",
                           "float64",
                           "string"])
    
    # AGD: All further operations are directly on the feats DataFrames
    avg = sd = pat = 0
    if (feats1.shape[0] * feats2.shape[0]) == 0:
        pat = 1
        if feats1.shape[0] > 0:
            avg = feats1.score.max()/2.0
            sd = np.std([0, feats1.score.max()])
        elif feats2.shape[0] > 0:
            avg = feats2.score.max()/2.0
            sd = np.std([0, feats2.score.max()])
        # else no motifs -- avg = sd = pat = 0
    else:
        index1 = feats1.score.idxmax()
        index2 = feats2.score.idxmax()
        strand1 = feats1.strand.iloc[index1]
        strand2 = feats2.strand.iloc[index2]
        pat = assign_motif_pattern(strand1, strand2)
        avg = np.mean( [feats1.score.max(), feats2.score.max()] )
        sd = np.std( [feats1.score.max(), feats2.score.max()] )

    lock.acquire()
    scores1[i] = avg
    scores2[i] = sd
    pattern[i] = pat
    lock.release()

    
def add_motif_pattern(train, Peak, opt):
    """
    This function is to add the motif pattern feature for interacting anchors in training data.
    Peak is the complete path for the table containing all these information for all motifs. The format is:
    chrom + start + end + strand + pvalue + score + phastCon
    Return a training data with added motif pattern feature
    """

    base1 = multiprocessing.Array(ctypes.c_double, train.shape[0])
    scores1 = np.ctypeslib.as_array(base1.get_obj())
    base2 = multiprocessing.Array(ctypes.c_double, train.shape[0])
    scores2 = np.ctypeslib.as_array(base2.get_obj())
    base3 = multiprocessing.Array(ctypes.c_int, train.shape[0])
    pattern = np.ctypeslib.as_array(base3.get_obj())
    
    # Create the multiprocessing thread pool
    pool = multiprocessing.Pool(processes = opt.procs,
                                initializer = _init_motifs,
                                initargs = (scores1, scores2, pattern))
    
    map_args = []
    for i in range(0,train.shape[0]):
        map_args.append((i, train, Peak, opt))
        
    pool.map(find_motif_pattern, map_args)
    pool.close()
    pool.join()
    
    train['motif_pattern'] = pd.Series(pattern, index = train.index)
    train['avg_motif_strength'] = pd.Series(scores1, index = train.index)
    train['std_motif_strength'] = pd.Series(scores2, index = train.index)
    return train


def add_anchor_conservation(train, chroms, Peak):
    """
    To add the feature of sequence conservation on anchors.
    Peak is the folder that contains the phastCon.wig files of all chroms.
    The name format of PhastCon of each chrom is chrXX.phastCons100way.wigFix
    """
    starts = {}
    cvg = {}

    ext = 20
    training = [] # [chrom_train_DF, chrom2_train_DF, ...]
    for chrom in chroms:
        chrom_train = train[train['chrom'] == chrom].copy()
        chrom_train.reset_index(inplace=True)
        print 'Read in phastCon in '+chrom+'...'

        # Read in the phastCon track
        cvg = [0]*GenomeData.hg19_chrom_lengths[chrom]
        phastCon = Peak+'/'+chrom+'.phastCons100way.wigFix'
        wiggle = open(phastCon,'r')
        for line in wiggle:
            if line[0] == 'f':
                i = 0
                start = int(line.strip().split(' ')[2].split('=')[1])
            else:
                signal = line.strip().split(' ')[0]
                if signal == 'NA':
                    signal = 0
                else:
                    signal = float(signal)
                cvg[start + i] = signal
                i += 1
        wiggle.close()

        AvgCons = []
        DevCons = []
        for index, row in chrom_train.iterrows():
            con1 = sum(cvg[(row['peak1']-ext): (row['peak1']+ext)])
            con2 = sum(cvg[(row['peak2']-ext): (row['peak2']+ext)])
            AvgCons.append((con1+con2)/2.0)
            DevCons.append(np.std([con1, con2]))
        chrom_train['avg_conservation'] = pd.Series(AvgCons)
        chrom_train['std_conservation'] = pd.Series(DevCons)
        training.append(chrom_train)

    new_train = pd.concat(training, ignore_index=True)

    return new_train


def add_local_feature(signal, train, BED):
    """
    This function is to calculate the signal values on summit positions (summit +/- 2kb)
    """
    sys.stderr.write("\t\tPreparing the local features of {} ...".format(str(signal))) # AGD: Use stderr and str.format
    extension = 2000
    fragment = 150 # This is the size of the ChIP fragment, usually it is 150.
    shift = fragment/2
    BED_reader = open(BED,'r')
    read_info = {}  # read_info = {'chrom':[peak1, peak2,...]}  actually 'start' here is the mid-point of a fragment

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
        peak1 = row['peak1']
        peak2 = row['peak2']

        count1 = bisect.bisect_right(read_info[chrom], peak1+extension) - bisect.bisect_left(read_info[chrom], peak1-extension)
        count2 = bisect.bisect_right(read_info[chrom], peak2+extension) - bisect.bisect_left(read_info[chrom], peak2-extension)

        RPKM1 = float(count1)/(float(read_number)*2*extension)*1000000000
        RPKM2 = float(count2)/(float(read_number)*2*extension)*1000000000

        RPKMs1.append((RPKM1+RPKM2)/2.0)
        RPKMs2.append(np.std([RPKM1, RPKM2]))

    signal1 = 'avg_'+str(signal)
    signal2 = 'std_'+str(signal)
    train[signal1] = pd.Series(RPKMs1, index = train.index)
    train[signal2] = pd.Series(RPKMs2, index = train.index)
    return train


def add_peak_feature(signal, train, BED, opt):
    """
    This function calculates signal values for narrowPeak features overlapping summit positions.
    Added by AGD, 1/26/2018
    """
    base1 = multiprocessing.Array(ctypes.c_double, train.shape[0])
    scores1 = np.ctypeslib.as_array(base1.get_obj())
    base2 = multiprocessing.Array(ctypes.c_double, train.shape[0])
    scores2 = np.ctypeslib.as_array(base2.get_obj())
    
    # Create the multiprocessing thread pool
    pool = multiprocessing.Pool(processes = opt.procs,
                                initializer = _init_peaks,
                                initargs = (scores1, scores2))

    map_args = []
    for i in range(0,train.shape[0]):
        map_args.append((i, train, BED, opt))

    pool.map(do_peak_feat_row, map_args)
    pool.close()
    pool.join()
    
    signal1 = 'avg_'+str(signal)
    signal2 = 'std_'+str(signal)
    train[signal1] = pd.Series(scores1, index = train.index)
    train[signal2] = pd.Series(scores2, index = train.index)
    return train


def do_peak_feat_row(map_args, def_param=(scores1,scores2)):
    """
    Loop definition for multithreading over table rows within add_peak_features.
    """
    (i, train, BED, opt) = map_args
    peaks = tb.open(BED)
    row = train.iloc[i]
    anchor1, anchor2 = prepare_anchors(row, opt.extension)

    feats1 = get_features(anchor1.chrom,
                          anchor1.start,
                          anchor1.end,
                          peaks,
                          ["chrom",
                           "chromStart",
                           "chromEnd",
                           "name",
                           "score",
                           "strand",
                           "signalValue",
                           "pValue",
                           "qValue",
                           "peak"],
                          ["string",
                           "int64",
                           "int64",
                           "string",
                           "float64",
                           "string",
                           "int64",
                           "float64",
                           "float64",
                           "int64"])
    feats2 = get_features(anchor2.chrom,
                          anchor2.start,
                          anchor2.end,
                          peaks,
                          ["chrom",
                           "chromStart",
                           "chromEnd",
                           "name",
                           "score",
                           "strand",
                           "signalValue",
                           "pValue",
                           "qValue",
                           "peak"],
                          ["string",
                           "int64",
                           "int64",
                           "string",
                           "float64",
                           "string",
                           "int64",
                           "float64",
                           "float64",
                           "int64"])

    score1 = choose_feat(feats1, "signalValue", opt)
    score2 = choose_feat(feats2, "signalValue", opt)
    lock.acquire()
    scores1[i] = (score1+score2)/2.0
    scores2[i] = np.std([score1, score2])
    lock.release()
    
    
def choose_feat(feats, col, opt):
    """
    Given a set of features and a selection criterion (in opt.collapse_peaks),
    choose the "best" row or aggregate over all rows for given column, and return 
    the chosen value.
    """
    if feats.shape[0] == 0:
        return 0
    if opt.collapse_peaks == "max":
        return feats[col].max()
    if opt.collapse_peaks == "min":
        return feats[col].min()
    if opt.collapse_peaks == "sum":
        return feats[col].sum()
    if opt.collapse_peaks == "avg":
        return feats[col].mean()


def add_gene_expression(train, Peak):
    """
    This function is to add the gene expression value of the looped region as a feature.The gene expression file's format is:
    gene_id   locus   value
    A1BG    chr19:coordinate1-coordiate2   1.31

    """
    exp_file = pd.read_table(Peak)
    gene_exp = {}  # {'chrom':{iv1:fpkm1,iv2:fpkm2...}}
    for index, row in exp_file.iterrows():
        gene = row['gene_id']
        region = row['locus']
        fpkm = row['value']
        chrom = region.split(':')[0]
        start = int(region.split(':')[1].split('-')[0])
        end = int(region.split(':')[1].split('-')[1])
        iv = HTSeq.GenomicInterval(chrom, start, end, '.')
        if chrom not in gene_exp.keys():
            gene_exp[chrom] = {}
        gene_exp[chrom][iv] = fpkm

    loop_expressions = []
    for index, row in train.iterrows():
        chrom = row['chrom']
        peak1 = row['peak1']
        peak2 = row['peak2']
        iv = HTSeq.GenomicInterval(chrom, peak1, peak2)
        loop_expression = 0
        for gene in gene_exp[chrom].keys():
            if gene.overlaps(iv):
                loop_expression += gene_exp[chrom][gene]
        loop_expressions.append(loop_expression)
    train['expression'] = pd.Series(loop_expressions, index = train.index)
    return train


def add_regional_feature_by_reads(signal, train, anchors, BED):
    """
    This function is to add the in-between and loop-flanking features for an interacting pair from raw reads BED files, whose format is:
    chrom+start+end.

    """
    print "\t\tPreparing the in-between and loop-flanking features of "+str(signal)+'...'
    BED = open(BED, 'r')
    shift = 75
    signal_dic = {}  # signal_dic = {"chrXX":[peak1, peak2, ...]} 'start' here are mid-point of one fragment
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
        peak1 = row['peak1']
        peak2 = row['peak2']

        index1 = anchors[chrom].index(peak1)
        index2 = anchors[chrom].index(peak2)
        if index1 != 0:
            up_motif = anchors[chrom][index1 - 1]
            up_count = bisect.bisect_right(signal_dic[chrom], peak1) - bisect.bisect_left(signal_dic[chrom], up_motif)
            up_strength = float(up_count)/float(abs(up_motif-peak1)*read_number)*1e+9
        else:
            up_strength = 0
        upstream.append(up_strength)
        if index2 != (len(anchors[chrom])-1):
            down_motif = anchors[chrom][index2 + 1]
            down_count = bisect.bisect_right(signal_dic[chrom], down_motif) - bisect.bisect_left(signal_dic[chrom], peak2)
            down_strength = float(down_count)/float(abs(down_motif-peak2)*read_number)*1e+9
        else:
            down_strength = 0
        downstream.append(down_strength)

        strength = 0
        count = bisect.bisect_right(signal_dic[chrom], peak2) - bisect.bisect_left(signal_dic[chrom], peak1)

        strength = float(count)/float(abs(peak2-peak1)*read_number)*1e+9
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
        parser.add_option('-e', '--extension', type=int, default=2000,
                          help='Size of the extesion window to append up and downstream of each anchor peak for signal calculation. Default 2000.')
        parser.add_option('-m', '--motif_extension', type=int, default=500,
                          help='Size of extension window to append when finding motif features. Default 500.')
        parser.add_option('-c', '--collapse_peaks', type='choice', choices=["max", "avg", "min", "sum"], default='max',
                          help='How to handle multiple overlapping peak features. Allowed values: max (use maximum score), avg (average all scores), min (use lowest score), sum (use sum of all scores). Default = max.')        

        
	(opt, args) = parser.parse_args(argv)
	if len(argv) < 6:
		parser.print_help()
        	sys.exit(1)

	start_time = time.time()
	chroms = GenomeData.hg19_chroms
	#chroms = ['chr1']
	train = pd.read_table(opt.training)
	train = train.sort_values(by=['chrom','peak1','peak2'], axis = 0, ascending=[1,1,1])
	info_table = pd.read_table(opt.info_table)
	anchors = {} # anchors = {'chr':set(summit1, summit2,)}
	
        # Generate the anchors pool from the anchors of positive loops
        sys.stderr.write("Preparing the anchor pool...\n")
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
		    train = add_anchor_conservation(train, chroms, Path)
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
