#!/usr/bin/python

"""
Combine RNA-seq biological replicates from ENCODE.
Input data are gene quantification files from the
ENCODE DCC portal, in tsv format. By default, will
use the pme_TPM column (0-based column 9) as the
signal. Output is two columns: gene_id, pme_TPM.
"""

import sys
import numpy as np
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog="Adam Diehl (Alan Boyle Lab, University of Michigan)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", nargs='+',
                        help="RNA-seq gene quantification files. Assumed to be gene quantification files from ENCODE, in tsv format. If multiple files are given, expression values will be averaged across all replicates. Also assumes that all files contain the same number of lines and that the same lines describe the same genes in all files.")
    parser.add_argument('-c', '--signal_column', type=int, default=9,
                        help="Column containing expression values to be averaged. Zero-based. Default = 9 (pme_TPM).")

    opt = parser.parse_args()

    # Read input files into pandas dataframes    
    files = []
    for input in opt.input:        
        files.append(pd.read_table(input,
                                   dtype={"gene_id": "string",
                                          "transcript_id(s)": "string",
                                          "length": "float",
                                          "effective_length": "float",
                                          "expected_count": "float",
                                          "TPM": "float",
                                          "FPKM": "float",
                                          "posterior_mean_count": "float",
                                          "posterior_standard_deviation_of_count": "float",
                                          "pme_TPM": "float",
                                          "pme_FPKM": "float",
                                          "TPM_ci_lower_bound": "float",
                                          "TPM_ci_upper_bound": "float",
                                          "FPKM_ci_lower_bound": "float",
                                          "FPKM_ci_upper_bound": "float"}))

    # Check lengths of each input
    if len(files)> 1:
        l = files[0].shape[0]
        for i in range(1, len(files)):            
            if files[i].shape[0] != l:
                sys.stderr.write("ERROR: Input files are of different lengths. Try --help.\n")
                exit(1)
                
    # Print the output file header
    print "gene_id\t{}".format(files[0].columns[opt.signal_column])
    
    for i in range(files[0].shape[0]):
        gene_id = files[0].iloc[i].gene_id
        expr = 0
        for j in range(len(files)):
            expr += files[j].iloc[i,opt.signal_column]

        print "{}\t{}".format(gene_id, expr/len(files))
        
