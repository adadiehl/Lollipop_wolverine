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
import GenomeData


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
                      help='Size of the extesion window to append up and downstream of each anchor peak for signal calculation. Default 2000 (as in original). Set to 0 to use actual element boundaries.')
    parser.add_option('-m', '--motif_extension', type=int, default=500,
                      help='Size of extension window to append when finding motif features. Default 500. Set to 0 to use actual element boundaries.')
    parser.add_option('-z', '--cons_extension', type=int, default=20,
                      help='Size of extension window to append when calculating conservation. Default 20. Set to 0 to use actual element boundaries.')
    parser.add_option('-a', '--collapse_peaks', type='choice', choices=["max", "avg", "min", "sum"], default='max',
                      help='How to handle multiple overlapping peak features. Allowed values: max (use maximum score), avg (average all scores), min (use lowest score), sum (use sum of all scores). Default = max.')
    parser.add_option('-f', '--ctcf_f', type='string', help='Tabix-indexed CTCF peaks file in narrowPeak format.')
    parser.add_option('-r', '--report_extension', dest="report_actual", action='store_false', default=True,
                      help='Report actual ChIP-seq peak boundaries in output instead of peak +- extension.')
    parser.add_option('-n', '--no_in_between_peaks', dest="in_between", action='store_false', default=True,
                      help='Do not include "in-between" peaks in training and predictions.')
    parser.add_option('-g','--no_flanking_peaks', dest="flanking",action='store_false', default=True,
                      help='Do not include "upstream" and "downstream" peaks in training and predictions.')
    
    (opt, args) = parser.parse_args(argv)
    if len(argv) < 8:
        parser.print_help()
        sys.exit(1)

    loop_clf = joblib.load(opt.clf)
    outfilename = opt.outdir+'/Lollipop_loops.txt'
    bedope_name = opt.outdir+'/Lollipop_loops.bedpe'
    loop_cvg = opt.outdir+'/Lollipop_loops_cvg.bedgraph'

    # Read in the signals table
    signal_table, signals = lib.load_signals_table(opt.info_table)

    # Load CTCF Summits
    sys.stderr.write("Preparing CTCF summits list...\n")
    summits, chroms, CTCF_ChIP = lib.prepare_bs_pool(opt.bs, GenomeData.hg19_chroms, ["chrY"])

    # Prepare reads from read-based sources
    sys.stderr.write('Preparing reads information...\n')
    read_info, read_numbers = lib.prepare_reads_info(signal_table)

    # Open output streams
    outfile = open(outfilename, 'w')
    bedope = open(bedope_name, 'w')
    cvg = HTSeq.GenomicArray('auto', stranded=False, typecode='i')

    for chrom in summits.keys():        
        sys.stderr.write("Preparing data for {}...\n".format(chrom))

        # Prepare a table of putative interactions within the upper and lower bounds for loop length
        sys.stderr.write("\tPreparing putative interactions...\n")
        data = lib.prepare_interactions(chrom,
                                        CTCF_ChIP[CTCF_ChIP.chrom == chrom],
                                        opt.ctcf_f, opt.proximal, opt.distal, opt)
        data = lib.prepare_features_for_interactions(data, summits, signal_table, read_info, read_numbers, opt)
        #sys.stderr.write("{}\n".format(data.head()))

        # Strip genomic coordinates and convert annotations to np.matrix form
        X = data.iloc[:,7:].as_matrix()

        sys.stderr.write("\tPredicting loops with the random forest classifier...\n")
        # Predict classes (loop or background) with the trained random forest classifier
        y = loop_clf.predict(X[:,:])
        # Calculate class probabilities with the trained random forest classifier 
        probas = loop_clf.predict_proba(X[:,:]) # probas is an array of shape [n_samples, n_classes]

        # Print results
        sys.stderr.write("\tWriting results...\n")
        for i in xrange(len(y)):
            if (y[i] != 0):
                outfile.write("{}\t{}\t{}\t{}\t{}\n".format(chrom,
                                                            int(data.iloc[i].peak1),
                                                            int(data.iloc[i].peak2),
                                                            probas[i,1],
                                                            y[i]))
                if not opt.report_actual:
                    bedope.write("{}\t{}\t{}\t{}\t{}\t{}\tNA\t{}\t.\t.\t1\n".format(chrom,
                                                                                    int(data.iloc[i,1]-opt.extension),
                                                                                    int(data.iloc[i,1]+opt.extension),
                                                                                    chrom,
                                                                                    int(data.iloc[i,2]-opt.extension),
                                                                                    int(data.iloc[i,2]+opt.extension),
                                                                                    probas[i,1]))
                else:
                    bedope.write("{}\t{}\t{}\t{}\t{}\t{}\tNA\t{}\t.\t.\t1\n".format(chrom,
                                                                                    int(data.iloc[i].start1),
                                                                                    int(data.iloc[i].end1),
                                                                                    chrom,
                                                                                    int(data.iloc[i].start2),
                                                                                    int(data.iloc[i].end2),
                                                                                    probas[i,1]))
                iv = HTSeq.GenomicInterval(chrom, int(data.iloc[i,1]), int(data.iloc[i,2]), '.')
                cvg[iv] += 1

    cvg.write_bedgraph_file(loop_cvg)
    outfile.close()
    bedope.close()



if __name__ == "__main__":
    main(sys.argv)
