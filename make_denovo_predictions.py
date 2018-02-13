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

#import multiprocessing
#import ctypes

def load_signals_table(info_table):
    """
    Read the signals table from disk.
    TO-DO: Is this really needed??
    """
    signal_table = pd.read_table(info_table)
    signals = []
    for index, row in signal_table.iterrows():
        if (row['Signal'] != 'Motif' and row['Signal'] != 'Gene expression' and row['Signal'] != 'PhastCon'):
            signal = row['Signal']
            signals.append(signal)
    return signal_table, signals


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
    parser.add_option('-e', '--extension', type=int, default=1000,
                      help='Size of the extesion window to append up and downstream of each anchor peak for signal calculation. Default 1000 (for 2000bp feature window). Set to 0 top use actual peak boundaries.')
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

    
    loop_clf = joblib.load(opt.clf)
    outfilename = opt.outdir+'/Lollipop_loops.txt'
    bedope_name = opt.outdir+'/Lollipop_loops.bedpe'
    loop_cvg = opt.outdir+'/Lollipop_loops_cvg.bedgraph'

    # Read in the signals table
    signal_table, signals = load_signals_table(opt.info_table)

    
    # TO-DO: Break this out into a multithreaded function
    sys.stderr.write("Preparing CTCF summits list...\n")
    # Returns a pandas DataFrame
    CTCF_ChIP = lib.read_narrowPeak(opt.bs)

    summits = {}
    for index, row in CTCF_ChIP.iterrows():
        chrom = row['chrom']
        # Assumes narrowPeak format!
        summit = row['chromStart']+row['peak']
        if chrom not in summits.keys():
            summits[chrom] = set()
        summits[chrom].add(summit)
    sys.stderr.write("Sorting CTCF summits...\n")
    for chrom in summits.keys():
        summits[chrom] = sorted(list(summits[chrom]))
    chroms = CTCF_ChIP.chrom.unique()

    sys.stderr.write('Preparing reads information...\n')
    read_info, read_numbers = lib.prepare_reads_info(signal_table)
    

    # Open output streams
    i = 0
    outfile = open(outfilename, 'w')
    bedope = open(bedope_name, 'w')
    cvg = HTSeq.GenomicArray('auto', stranded=False, typecode='i')

    for chrom in chroms:

        sys.stderr.write("Preparing data for {}...\n".format(chrom))

        # Prepare a table of putative interactions within the upper and lower bounds for loop length
        sys.stderr.write("\tPreparing putative interactions...\n")
        data = lib.prepare_interactions(chrom,
                                        CTCF_ChIP[CTCF_ChIP.chrom == chrom],
                                        opt.ctcf_f, opt.proximal, opt.distal, opt)
        data = lib.prepare_features_for_interactions(data, summits, signal_table, read_info, read_numbers, opt)

        # Strip genomic coordinates and convert annotations to np.matrix form
        X = data.iloc[:,12:].as_matrix()

        sys.stderr.write("\tPredicting loops with the random forest classifier...\n")
        # Predict classes (loop or background) with the trained random forest classifier 
        y = loop_clf.predict(X[:,:])
        # Calculate class probabilities with the trained random forest classifier
        probas = loop_clf.predict_proba(X[:,:]) # probas is an array of shape [n_samples, n_classes]

        for i in xrange(len(y)):
            if (y[i] != 0):
                outfile.write("{}\t{}\t{}\t{}\t{}\n".format(chrom,
                                                            int(data.iloc[i].peak1),
                                                            int(data.iloc[i].peak2),
                                                            probas[i,1],
                                                            y[i]))
                bedope.write("{}\t{}\t{}\t{}\t{}\t{}\tNA\t{}\t.\t.\t1\n".format(chrom,
                                                                                int(data.iloc[i,1]),
                                                                                int(data.iloc[i,2]),
                                                                                chrom,
                                                                                int(data.iloc[i,4]),
                                                                                int(data.iloc[i,5]),
                                                                                probas[i,1]))
                iv = HTSeq.GenomicInterval(chrom, int(data.iloc[i].peak1), int(data.iloc[i].peak2), '.')
                cvg[iv] += 1


    cvg.write_bedgraph_file(loop_cvg)
    outfile.close()
    bedope.close()



if __name__ == "__main__":
    main(sys.argv)
