#!/usr/bin/python
# AGD, 1/29/2018 -- Added shebang line.

# Author: Yan Kai, with modifications by Adam Diehl (AGD) noted
# Date: May 29th, 2017
# This script is to prepare the high-confidence positive and negative long-range interactions among CTCF binding sites, as revealed from ChIP-Seq or ChIA-PET.
# The input to generate positive data is the loops determined by CTCF ChIA-PET experiment.
# The negative data are randomly sampled from CTCF binding sites.

import sys, re, os
import GenomeData
import numpy as np
import bisect
import math
import pandas as pd
import HTSeq
from optparse import OptionParser
from lib import read_narrowPeak

import multiprocessing
import ctypes

"""
Global Variables
"""
i_right = []
dummy = [] # ...because i_right list isn't being passed properly to the innner function on its own!
loops = []

"""
Function definitions
"""

def _init_neg(ir, dummy):
    global i_right
    i_right = ir

def find_negative_interactions(bs_pool, true_loops, less_sig_loops, hic_loops, opt):
    """
    Find negative interaction loops.
    """
    negative_interactions = pd.DataFrame(columns=['chrom1',
                                                  'start1',
                                                  'end1',
                                                  'chrom2',
                                                  'start2',
                                                  'end2',
                                                  'name',
                                                  'score',
                                                  'strand1',
                                                  'strand2',
                                                  'peak1',
                                                  'peak2',
                                                  'response',
                                                  'length'])

    """
    Multiprocessing across chromosomes...
    """
    pool = multiprocessing.Pool(processes = opt.procs)
    map_args = []
    for chrom in true_loops.keys():        
        map_args.append((bs_pool, chrom, true_loops, less_sig_loops, hic_loops, opt))
        #negative_interactions = negative_interactions.append(do_neg_interactions_for_chrom((bs_pool, chrom, true_loops, less_sig_loops, hic_loops, opt)))
        
    negative_interactions = negative_interactions.append(pool.map(do_neg_interactions_for_chrom, map_args))
    pool.close()
    pool.join()

    return negative_interactions

def do_neg_interactions_for_chrom(map_args):
    """
    Find all negative interactions on a single chromosome in
    a multiprocessing context.
    """
    (bs_pool, chrom, true_loops, less_sig_loops, hic_loops, opt) = map_args    
    ret = pd.DataFrame(columns = ['chrom1',
                                  'start1',
                                  'end1',
                                  'chrom2',
                                  'start2',
                                  'end2',
                                  #'name',
                                  #'score',
                                  #'strand1',
                                  #'strand2',
                                  'peak1',
                                  'peak2',
                                  #'response',
                                  'length'])

    for i_left in xrange(bs_pool[chrom].shape[0]-1):
        row_l = bs_pool[chrom].iloc[i_left]
        df = pd.DataFrame(bs_pool[chrom].iloc[i_left+1:,[0,1,2,9]])
        df.columns = ['chrom2',
                      'start2',
                      'end2',
                      'peak2']
        df['chrom1'] = row_l['chrom']
        df['start1'] = row_l['chromStart']
        df['end1'] = row_l['chromEnd']
        df['peak1'] = row_l.chromStart + row_l.peak
        df.peak2 = df.start2 + df.peak2
        df['length'] = df.peak2 - df.peak1
        
        # Pre-filter by loop length
        df = df[ (df.length >= opt.min_loop_size) &
                 (df.length <= opt.max_loop_size)]
        if df.shape[0] == 0:
            continue

        # Filter by overlaps with ChIA-PET and Hi-C        
        for j in xrange(df.shape[0]):
            good = False
            if (df.iloc[j].peak1, df.iloc[j].peak2) not in true_loops[chrom]:
                if df.iloc[j].chrom1 in less_sig_loops.keys():
                    if (df.iloc[j].peak1, df.iloc[j].peak2) not in less_sig_loops[chrom]:
                        if df.iloc[j].chrom1 in hic_loops.keys():
                            if (df.iloc[j].peak1, df.iloc[j].peak2) not in hic_loops[chrom]:
                                good = True
                else:
                    if df.iloc[j].chrom1 in hic_loops.keys():
                        if (df.iloc[j].peak1, df.iloc[j].peak2) not in hic_loops[chrom]:
                            good = True
            if good:
                ret = ret.append(df.iloc[j])

    ret['name'] = 'NA'
    ret['score'] = 0
    ret['strand1'] = '.'
    ret['strand2'] = '.'
    ret['response'] = 0
    return ret


def find_summits_in_anchors(anchor, chrom_summits):
    """
    anchor = HTSeq.iv
    chrom_summits = []  # list of summit position on one chrom
    """
    chrom = anchor.chrom
    overlapped = 0
    for summit in chrom_summits.peak:
        pos = HTSeq.GenomicPosition(chrom, summit,'.')
        if pos.overlaps(anchor):
            ans = summit
            overlapped = 1
            break
    if overlapped == 0:
        ans = (anchor.start + anchor.end)/2
    return str(ans)


def prepare_bs_pool(peak_f, chroms):
    """
    Prepare the ChIP-seq binding site pool.
    bs_pool = {'chrom':[summit1, summit2,...]} 
    """
    
    peak = read_narrowPeak(peak_f)
    bs_pool = {}
    for chrom in peak.chrom.unique():
        if chrom in chroms and chrom != 'chrY':
            bs_pool[chrom] = peak[peak['chrom'] == chrom].sort_values(by=['chromStart', 'chromEnd'], ascending=[1,1])
    return bs_pool


def read_hic(hic_f, bs_pool):
    """
    Read in Hi-C data.
    """
    hic = pd.read_table(hic_f)
    hic_loops = {}
    for index, row in hic.iterrows():
        chrom = row['chrom']
        anchor1 = HTSeq.GenomicInterval(chrom, row['start1']-1000, row['start1']+1000, '.')
        anchor2 = HTSeq.GenomicInterval(chrom, row['start2']-1000, row['start2']+1000,'.')
        if chrom not in hic_loops.keys():
            hic_loops[chrom] = []
        anchor1_summit = find_summits_in_anchors(anchor1, bs_pool[chrom])
        anchor2_summit = find_summits_in_anchors(anchor2, bs_pool[chrom])
        hic_loops[chrom].append((anchor1_summit,anchor2_summit))
    return hic_loops


def find_positive_interactions(chiapet, hic_loops, bs_pool, chroms, outfile, opt):
    """
    Find positive interactions in the ChIA-PET data and prepare the
    true_loops and less_sig_loops lists.
    
    true_loops = {'chrXX':[(summit1,summit2),(summit1,summit2)...]} 
    less_sig_loops = {'chrXX':[(summit1,summit2),(summit1,summit2)...]}
    To ensure negative loops are not less significant true loops.

    TO-DO: Make Hi-C data optional
    """
    NumPos = 0
    true_loops = {}
    less_sig_loops = {}

    chia = pd.read_table(chiapet)
    for index, row in chia.iterrows():
        IAB = row['IAB']
        FDR = row['FDR']
        if (row['chrom1'] == row['chrom2'] and row['chrom1'] in chroms and row['chrom1'] != 'chrY'):
            chrom = row['chrom1']
            anchor1 = HTSeq.GenomicInterval(chrom, row['start1'], row['end1'],'.')
            anchor2 = HTSeq.GenomicInterval(chrom, row['start2'], row['end2'],'.')
            # Get the summit position of the anchors
            if chrom in bs_pool.keys():
                anchor1_summit = find_summits_in_anchors(anchor1, bs_pool[chrom])
                anchor2_summit = find_summits_in_anchors(anchor2, bs_pool[chrom])
            else:
                anchor1_summit = 'NaN'
                anchor2_summit = 'NaN'
        # distance is the genomic length between the two motifs. To focus on long-range interactions,
        # we required that distance >= 10kb and <= 1m
        if (anchor1_summit != 'NaN' and anchor2_summit != 'NaN'):
            if (int(anchor1_summit) > int(anchor2_summit)):
                temp = anchor1_summit
                anchor1_summit = anchor2_summit
                anchor2_summit = temp
            distance = int(anchor2_summit) - int(anchor1_summit)
            
            if (distance >= opt.min_loop_size and distance <= opt.max_loop_size):
                if (IAB >= 2 and FDR <= 0.05):
                    NumPos += 1
                    if chrom not in true_loops.keys():
                        true_loops[chrom] = []

                    true_loops[chrom].append((int(anchor1_summit), int(anchor2_summit)))
                    outfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(chrom,
                                                                                                    row['start1'],
                                                                                                    row['end1'],
                                                                                                    chrom,
                                                                                                    row['start2'],
                                                                                                    row['end2'],
                                                                                                    'NA',
                                                                                                    float(1),
                                                                                                    '.',
                                                                                                    '.',
                                                                                                    anchor1_summit,
                                                                                                    anchor2_summit,
                                                                                                    1,
                                                                                                    distance))
                else:
                    if chrom not in less_sig_loops.keys():
                        less_sig_loops[chrom] = []
                    less_sig_loops[chrom].append((int(anchor1_summit), int(anchor2_summit)))

    return NumPos, true_loops, less_sig_loops
    
    
def main(argv):
    parser = OptionParser()
    parser.add_option("-p", "--peak", action="store", type="string", dest="peak", metavar="<file>", help="CTCF ChIP-seq peaks in narrowPeak format")
    parser.add_option("-a", "--chiapet", action="store", type="string", dest="chiapet", metavar="<file>", help="the interaction file from ChIA-PET experiment")
    parser.add_option("-c", "--hic", action="store", type="string", dest="hic", metavar="<file>", help="the CTCF interactions identified by hic data")
    parser.add_option("-o", "--train", action="store", type="string", dest="training", metavar="<file>", help="the resulting file with positive and sampled negative interactions for training")
    parser.add_option('-l', '--min_loop_size', type=int, default=10000,
                      help="Minimum loop size.")
    parser.add_option('-u', '--max_loop_size', type=int, default=1000000,
                      help="Maximum loop size.")
    parser.add_option('-r', '--ratio', type=int, default=5,
                      help="Ratio of negative to positive interactions. Default 5.")
    parser.add_option('-t', '--procs', type=int, default=1,
                      help='Number of processors to use. Default 1.')

    (opt, args) = parser.parse_args(argv)
    if len(argv) < 8:
        parser.print_help()
        sys.exit(1)

    chroms = GenomeData.hg19_chroms

    # Preparethe binding site pool
    sys.stderr.write("Preparing the binding site pool...\n")
    bs_pool = prepare_bs_pool(opt.peak, chroms)

    # HiC loops are used to ensure that the randomly generated negative
    # loops are not true loops identified in HiC.
    sys.stderr.write("Reading the Hi-C data...\n")
    hic_loops = read_hic(opt.hic, bs_pool)

    # Open the output stream
    outfile = open(opt.training,'w')
    # Print the header
    outfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format('chrom1',
                                                                                    'start1',
                                                                                    'end1',
                                                                                    'chrom2',
                                                                                    'start2',
                                                                                    'end2',
                                                                                    'name',
                                                                                    'score',
                                                                                    'strand1',
                                                                                    'strand2',
                                                                                    'peak1',
                                                                                    'peak2',
                                                                                    'response',
                                                                                    'length'))

    # Counter for the number of positive loops
    sys.stderr.write("Preparing positive interactions...\n")
    NumPos, true_loops, less_sig_loops = find_positive_interactions(opt.chiapet,
                                                                     hic_loops,
                                                                     bs_pool,
                                                                     chroms,
                                                                     outfile, opt)
    sys.stderr.write("Found a total of {} significant positive loops.\n".format(NumPos))
   
    # Generate the negative interactions pool
    sys.stderr.write("Preparing negative interactions\n")
    negative_interactions = find_negative_interactions(bs_pool, true_loops, less_sig_loops, hic_loops, opt)
    sys.stderr.write("There are {} negative loops in total.\n".format(negative_interactions.shape[0]))


    # The nuber of negative interactions to select. If this exceeds the total
    # number found, use all negative interactions and throw a warning.
    NumNeg = NumPos * opt.ratio
    if NumNeg >= negative_interactions.shape[0]:
        sys.stderr.write("The total number of negative interactions is less than {} times the number of positive loops. Using all interactions.\n".format(opt.ratio))
        NumNeg = negative_interactions.shape[0]
    else:
        sys.stderr.write("Randomly selecting {} negative loops...\n".format(NumNeg))
        
    selected_neg = sorted(np.random.choice(range(0,negative_interactions.shape[0]), NumNeg, replace=False))
    for idx, iv in negative_interactions.iloc[selected_neg].iterrows():
        outfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(iv.chrom1,
                                                                                        int(iv.start1),
                                                                                        int(iv.end1),
                                                                                        iv.chrom2,
                                                                                        int(iv.start2),
                                                                                        int(iv.end2),
                                                                                        iv['name'],
                                                                                        float(iv.score),
                                                                                        iv.strand1,
                                                                                        iv.strand2,
                                                                                        int(iv.peak1),
                                                                                        int(iv.peak2),
                                                                                        int(iv.response),
                                                                                        int(iv.length)))
        
    outfile.close()

if __name__ == "__main__":
	main(sys.argv)
