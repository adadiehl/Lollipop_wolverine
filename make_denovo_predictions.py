#!/usr/bin/python
# AGD, 1/29/2018: Added shebang line

# Modifications by Adam Diehl (AGD) as noted.

import re,os,sys
from optparse import OptionParser
from sklearn import svm
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import HTSeq
import lib


def main(argv):
    parser = OptionParser()
    parser.add_option("-b", "--bs", action="store", type="string", dest="bs", metavar="<file>", help="the CTCF ChIP-Seq peak file")
    parser.add_option("-t", "--table", action="store", type="string", dest="info_table", metavar="<file>", help="the infomation table contains the paths of necessary files.")
    parser.add_option("-c", "--clf", action="store", type="string", dest="clf", metavar="<file>", help="The trained classifier for identifying the interacting loop pairs")
    parser.add_option("-o", "--outdir", action="store", type="string", dest="outdir", help="The directory for output files, predicted loops, etc")
    parser.add_option('-p', '--procs', type=int, default=1,
                      help='Number of processors to use. Default 1.')
    parser.add_option('-u', '--proximal', type=int, default=10000,
                      help='Minimum distance between upstream and downstream loop anchors. Default 10,000.')
    parser.add_option('-d', '--distal', type=int, default=1e+6,
                      help='Maximum distance between upstream and downstream loop anchors. Default 1e+6.')
    parser.add_option('-e', '--extension', type=int, default=2000,
                      help='Size of the extesion window to append up and downstream of each anchor peak for signal calculation. Default 2000 (as in original).')
    parser.add_option('-m', '--motif_extension', type=int, default=500,
                      help='Size of extension window to append when finding motif features. Default 500.')
    parser.add_option('-z', '--cons_extension', type=int, default=20,
                      help='Size of extension window to append when calculating conservation. Default 20.')
    parser.add_option('-a', '--collapse_peaks', type='choice', choices=["max", "avg", "min", "sum"], default='max',
                      help='How to handle multiple overlapping peak features. Allowed values: max (use maximum score), avg (average all scores), min (use lowest score), sum (use sum of all scores). Default = max.')
    parser.add_option('-f', '--ctcf_f', type='string', help='Tabix-indexed CTCF peaks file in narrowPeak format.')
    

    (opt, args) = parser.parse_args(argv)
    if len(argv) < 8:
        parser.print_help()
        sys.exit(1)

    signal_table = pd.read_table(opt.info_table)

    loop_clf = joblib.load(opt.clf)
    outfilename = opt.outdir+'/Lollipop_loops.txt'
    bedope_name = opt.outdir+'/Lollipop_loops.bedpe'
    loop_cvg = opt.outdir+'/Lollipop_loops_cvg.bedgraph'
    motif_length = 18


    signals = []
    for index, row in signal_table.iterrows():
        if (row['Signal'] != 'Motif' and row['Signal'] != 'Gene expression' and row['Signal'] != 'PhastCon'):
            signal = row['Signal']
            signals.append(signal)


    CTCF_ChIP = pd.read_table(opt.bs , header=None)
    summits = {}
    for index, row in CTCF_ChIP.iterrows():
        chrom = row[0]
        start = row[1]
        end = row[2]
        summit = (start + end)/2
        if chrom not in summits.keys():
            summits[chrom] = set()
        summits[chrom].add(summit)
    for chrom in summits.keys():
        summits[chrom] = list(summits[chrom])

    chroms = summits.keys()

    print 'Preparing reads information...'
    read_info, read_numbers = lib.prepare_reads_info(signal_table)

    distance_distal = 1e+6
    distance_proximal = 10000  # We focus on long-range interactions between [10kb,1mb]


    raw_features = (signals, read_info, read_numbers)
    i = 0
    outfile = open(outfilename, 'w')
    bedope = open(bedope_name, 'w')
    cvg = HTSeq.GenomicArray('auto', stranded=False, typecode='i')

    for chrom in summits.keys():
        print "Preparing data for "+chrom+'...'
        summits[chrom] = sorted(summits[chrom])
        data = pd.DataFrame(columns=['chrom','peak1','peak2','length'])
        for i in xrange(len(summits[chrom])-1):
            data = lib.prepare_interactions(data, chrom, i, summits[chrom], distance_distal, distance_proximal)
        """
        print "Preparing motif orientation pattern and strength..."
        data = lib.add_motif_pattern(data, signal_table.iloc[0,2])
        print 'Preparing sequence conservation feature...'
        data = lib.add_anchor_conservation(data, chrom, signal_table.iloc[1,2])

        data = lib.prepare_features_for_interactions(data, summits, signal_table, raw_features)
        """
        data = lib.prepare_features_for_interactions(data, summits, signal_table, read_info, read_numbers, opt)
        
        X = data.iloc[:,3:].as_matrix()
        y = loop_clf.predict(X[:,:])
        probas = loop_clf.predict_proba(X[:,:]) # probas is an array of shape [n_samples, n_classes]
        for i in xrange(len(y)):
            if (y[i] != 0):
                outline = chrom+'\t'+str(int(data.iloc[i,1]))+'\t'+str(int(data.iloc[i,2]))+'\t'+str(probas[i,1])+'\t'+str(y[i])+'\n'
                outfile.write(outline)
                outline1 = chrom+'\t'+str(int(data.iloc[i,1]))+'\t'+str(int(data.iloc[i,1])+motif_length)+'\t'+chrom+'\t'+str(int(data.iloc[i,2]))+'\t'+str(int(data.iloc[i,2])+motif_length)+'\t'+'NA'+'\t'+                str(probas[i,1])+'\t'+'.'+'\t'+'.'+'\t'+str(1)+'\n'
                bedope.write(outline1)
                iv = HTSeq.GenomicInterval(chrom, int(data.iloc[i,1]), int(data.iloc[i,2]), '.')
                cvg[iv] += 1


    cvg.write_bedgraph_file(loop_cvg)
    outfile.close()
    bedope.close()



if __name__ == "__main__":
    main(sys.argv)
