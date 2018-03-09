#!/usr/bin/python

"""
Given a gene expression table and gff3 file containing
gene annotations, produce a bed-format file with the
location, name, and expression level of the genes.
"""

import sys
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog="Adam Diehl (Alan Boyle Lab, University of Michigan)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("expr_file",
                        help="RNA-seq quantifications. Two columns, tab-delimited: gene_id, expr.")
    parser.add_argument("gff_file",
                        help="GFF3 gene models, from GENCODE. Any gene_id not found in this file will be silently dropped from the output.")

    opt = parser.parse_args()

    # Prepare the gene info table
    gff_f = pd.read_table(opt.gff_file,
                          names=["chrom",
                                 "cat",
                                 "type",
                                 "chromStart",
                                 "chromEnd",
                                 "c1",
                                 "strand",
                                 "c2",
                                 "params"],
                          dtype={"chrom": "string",
                                 "cat": "string",
                                 "type": "string",
                                 "chromStart": "int64",
                                 "chromEnd": "int64",
                                 "c1": "string",
                                 "strand": "string",
                                 "c2": "string",
                                 "params": "string"},
                          comment='#')

    gene_ids = []
    gene_names = []
    for idx, row in gff_f.iterrows():
        params = dict(item.split("=") for item in row.params.split(";"))
        gene_ids.append(params["gene_id"])
        gene_names.append(params["gene_name"])
        
    gff_f = gff_f[gff_f.type == "gene"].iloc[:,[0,3,4,6]]
    gff_f["gene_id"] = pd.Series(gene_ids)
    gff_f["gene_name"] = pd.Series(gene_names)

    # Associate rows in the expression data with entries in the gene info table
    expr_f = pd.read_table(opt.expr_file,
                           dtype={"gene_id": "string",
                                  "pme_TPM": "float"})
    for idx, line in expr_f.iterrows():
        gene_info = gff_f[gff_f.gene_id == line.gene_id]
        if gene_info.shape[0] > 0:
            print "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(gene_info.iloc[0].chrom,
                                                      gene_info.iloc[0].chromStart,
                                                      gene_info.iloc[0].chromEnd,
                                                      gene_info.iloc[0].gene_name,
                                                      line.gene_id,
                                                      gene_info.iloc[0].strand,
                                                      line.pme_TPM)
            
