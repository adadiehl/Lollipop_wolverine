import sys, re, os
import numpy as np
import bisect
import collections
import pandas as pd
import HTSeq
import operator
import GenomeData

"""
Added by AGD

"""

import multiprocessing
import tabix as tb
import ctypes
from scipy.stats import variation

"""
Global Variables

"""
pattern = []
scores1 = []
scores2 = []
lock = multiprocessing.Lock()


"""
Function definitions

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


def read_narrowPeak(peaks_f):
    """
    Read a narrowPeak file into a pandas DataFrame
    """
    peaks = pd.read_table(peaks_f ,
                          header=None,
                          names=["chrom",
                                 "chromStart",
                                 "chromEnd",
                                 "name",
                                 "score",
                                 "strand",
                                 "signalValue",
                                 "pValue",
                                 "qValue",
                                 "peak"],
                          dtype={"chrom": "string",
                                 "chromStart": "int64",
                                 "chromEnd": "int64",
                                 "name": "string",
                                 "score": "int64",
                                 "strand": "string",
                                 "signalValue": "float64",
                                 "pValue": "float64",
                                 "qValue": "float64",
                                 "peak": "int64"})
    return peaks


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


def prepare_bs_pool(bs, chroms, exclude_chroms):
    """
    Prepare the ChIP-seq binding site pool.
    bs_pool = {'chrom':[summit1, summit2,...]}
    """
    peak = read_narrowPeak(bs)
    bs_pool = {}    
    for index, row in peak.iterrows():
        # Assumes narrowPeak format!
        chrom = row['chrom']
        summit = row['chromStart'] + row['peak']
        if (chrom not in bs_pool.keys() and
            chrom in chroms and
            chrom not in exclude_chroms):
            bs_pool[chrom] = set()
        bs_pool[chrom].add(summit)
    for chrom in bs_pool.keys():
        bs_pool[chrom] = sorted(list(bs_pool[chrom]))
    chroms = bs_pool.keys()
    return bs_pool, chroms, peak


def prepare_anchors_pool(bs, chroms, exclude_chroms):
    """
    Prepare the ChIP-seq anchors site pool.
    anchors_pool = {'chrom':[(start1, end1, summit1), (start2, end2, summit2),...]}
    """
    peak = read_narrowPeak(bs)
    bs_pool = {}
    for index, row in peak.iterrows():
        # Assumes narrowPeak format!
        chrom = row['chrom']
        summit = row['chromStart'] + row['peak']
        if (chrom not in bs_pool.keys() and
            chrom in chroms and
            chrom not in exclude_chroms):
            bs_pool[chrom] = set()
        bs_pool[chrom].add((row['chromStart'],
                            row['chromEnd'],
                            summit))
    for chrom in bs_pool.keys():
        bs_pool[chrom] = sorted(list(bs_pool[chrom]))
    chroms = bs_pool.keys()
    return bs_pool, chroms, peak
                                                                                            


def prepare_anchors(row, ext):
    """
    Added by AGD, 1/26/2018
    Prepare a set of anchors from a pair of anchor summits.

    Inputs:
        row = a row from the training data table
        ext = the number of bp to extend the peak up and downstream.
    """
    chrom = row['chrom']
    start1 = row['peak1'] - ext
    start2 = row['peak2'] - ext
    end1 = row['peak1'] + ext
    end2 = row['peak2'] + ext
    anchor1 = HTSeq.GenomicInterval(chrom, start1, end1, '.')
    anchor2 = HTSeq.GenomicInterval(chrom, start2, end2, '.')
    return anchor1, anchor2
                                                        

def find_motif_pattern(map_args, def_param=(scores1, scores2, pattern)):
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
                                                                                    

def choose_feat(feats, col, collapse):
    """
    Given a set of features and a selection criterion (in opt.collapse_peaks)
    choose the "best" row or aggregate over all rows for given column, and return
    the chosen value.
    """
    if feats.shape[0] == 0:
        return 0
    if collapse == "max":
        return feats[col].max()
    if collapse == "min":
        return feats[col].min()
    if collapse == "sum":
        return feats[col].sum()
    if collapse == "avg":
        return feats[col].mean()


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


def add_peak_inbetween(signal, train, BED, opt):
    """
    This function adds "in-between" signals for peak features.
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
        
    pool.map(do_peak_inbetween_row, map_args)
    pool.close()
    pool.join()
    
    signal1 = "{}_inbetween".format(signal)
    train[signal1] = pd.Series(scores1, index = train.index)
    return train


def add_peak_flanking(signal, train, BED, anchors, opt):
    """
    This function adds "upstream" and "downstream" signals for peak features.
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
        map_args.append((i, train, BED, anchors, opt))

    pool.map(do_peak_flanking_row, map_args)
    pool.close()
    pool.join()
    
    signal1 = "{}_upstream".format(signal)
    signal2 = "{}_downstream".format(signal)
    train[signal1] = pd.Series(scores1, index = train.index)
    train[signal2] = pd.Series(scores2, index = train.index)
    return train


def add_gene_expr(signal, train, BED, opt):
    """
    This function adds gene expression signal between peak features.
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
    
    pool.map(do_gene_expr_row, map_args)
    pool.close()
    pool.join()
    
    signal1 = "{}_avg".format(signal)
    signal2 = "{}_variation".format(signal)
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
                           "int64",
                           "string",
                           "float64",
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
                           "int64",
                           "string",
                           "float64",
                           "float64",
                           "float64",
                           "int64"])
    
    score1 = choose_feat(feats1, "signalValue", opt.collapse_peaks)
    score2 = choose_feat(feats2, "signalValue", opt.collapse_peaks)
    lock.acquire()
    scores1[i] = (score1+score2)/2.0
    scores2[i] = np.std([score1, score2])
    lock.release()


def do_peak_inbetween_row(map_args, def_param=(scores1,scores2)):
    """
    Loop definition for multithreading over table rows within add_peak_inbetween.
    """
    (i, train, BED, opt) = map_args
    peaks = tb.open(BED)
    row = train.iloc[i]

    # Get all peak features between the anchor summits
    feats1 = get_features(row.chrom,
                          row.peak1,
                          row.peak2,
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
                           "int64",
                           "string",
                           "float64",
                           "float64",
                           "float64",
                           "int64"])
    lock.acquire()
    # Using "sum" to aggregate the scores should roughly approximate the read-based case
    scores1[i] = choose_feat(feats1, "signalValue", "sum")
    #scores2[i] = np.std(list(feats1.signalValue))
    lock.release()


def do_peak_flanking_row(map_args, def_param=(scores1,scores2)):
    """
    Loop definition for multithreading over table rows within add_peak_features.
    """
    (i, train, BED, anchors, opt) = map_args
    peaks = tb.open(BED)
    row = train.iloc[i]
    index1 = anchors[row.chrom].index(row.peak1)
    index2 = anchors[row.chrom].index(row.peak2)

    if index1 > 0:
        feats1 = get_features(row.chrom,
                              anchors[row.chrom][index1-1],
                              row.peak1,
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
                               "int64",
                               "string",
                               "float64",
                               "float64",
                               "float64",
                               "int64"])
        score1 = choose_feat(feats1, "signalValue", "sum")
    else:
        score1 = 0
    if index2 < len(anchors[row.chrom])-1:
        feats2 = get_features(row.chrom,
                              row.peak2,
                              anchors[row.chrom][index2+1],
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
                               "int64",
                               "string",
                               "float64",
                               "float64",
                               "float64",
                               "int64"])
        score2 = choose_feat(feats2, "signalValue", "sum")
    else:
        score2 = 0
    lock.acquire()
    scores1[i] = score1
    scores2[i] = score2
    lock.release()


def do_gene_expr_row(map_args, def_param=(scores1,scores2)):
    """
    Loop definition for multithreading over table rows within add_gene_expr.
    """
    (i, train, BED, opt) = map_args
    peaks = tb.open(BED)
    row = train.iloc[i]
    
    # Get all peak features between the anchor summits
    feats1 = get_features(row.chrom,
                          row.peak1,
                          row.peak2,
                          peaks,
                          ["chrom",
                           "chromStart",
                           "chromEnd",
                           "name",
                           "id",
                           "strand",
                           "signalValue"],
                          ["string",
                           "int64",
                           "int64",
                           "string",
                           "string",
                           "string",
                           "float64"])
    lock.acquire()
    scores1[i] = choose_feat(feats1, "signalValue", "avg")
    scores2[i] = variation(list(feats1.signalValue))
    lock.release()
                                                
    
def add_bigWig_feature(train, Peak, opt):
    """                                                                                                           
    Adds scores from bigWig features.                                                                             
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
        map_args.append((i, train, Peak, opt))

    pool.map(get_bigWig_scores, map_args)
    pool.close()
    pool.join()
    
    train['avg_conservation'] = pd.Series(scores1)
    train['std_conservation'] = pd.Series(scores2)

    return train


def get_bigWig_scores(map_args, def_param=(scores1,scores2)):
    """                                                                                                           
    Inner loop for multithreading over bigWig score features.                                                     
    """
    (i, train, Peak, opt) = map_args
    bw = pyBigWig.open(Peak)
    row = train.iloc[i]
    anchor1, anchor2 = prepare_anchors(row, opt.cons_extension)
    con1 = sum(bw.values(anchor1.chrom,
                         anchor1.start,
                         anchor1.end))
    con2 = sum(bw.values(anchor2.chrom,
                         anchor2.start,
                         anchor2.end))
    lock.acquire()
    scores1[i] = (con1+con2)/2.0
    scores2[i] = np.std([con1, con2])
    lock.release()
    
    
def prepare_interactions(chrom, anchors, BED, proximal, distal, opt):
    """
    This function is to prepare the potential interactions for all peaks on one chromosome
    with all the downnstream peaks on the same chromosome. Assumes that anchors is a pandas
    dataframe read in by read_narrowPeak, and BED is the Tabix-indexed narrowPeak file
    containing the same peaks stored in the anchors table. Returns a pandas dataFrame
    following bedpe column conventions.
    """
    data = pd.DataFrame(columns=['chrom',
                                 'start1',
                                 'end1',
                                 'start2',
                                 'end2',
                                 'peak1',
                                 'peak2',
                                 'length'])
    for idx, anchor in anchors.iterrows():
        peaks = tb.open(BED)
        feats = get_features(anchor.chrom,
                             anchor.chromStart + anchor.peak + proximal,
                             anchor.chromStart + anchor.peak + distal,
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
                              "int64",
                              "string",
                              "float64",
                              "float64",
                              "float64",
                              "int64"])
        
        data1 = pd.DataFrame(columns=['chrom',
                                      'start1',
                                      'end1',
                                      'start2',
                                      'end2',
                                      'peak1',
                                      'peak2',
                                      'length'])
        data1['start2'] = feats.chromStart
        data1['end2'] = feats.chromEnd
        data1['peak2'] = feats.chromStart + feats.peak
        data1['peak1'] = anchor.chromStart + anchor.peak
        data1['chrom'] =  chrom
        data1['start1'] = anchor.chromStart
        data1['end1'] = anchor.chromEnd
        data1['length'] = data1['peak2'] - data1['peak1']
        
        data = pd.concat([data,data1],ignore_index=True)

    return data


def prepare_features_for_interactions(data, summits, signal_table, read_info, read_numbers, opt):
    """
    data is a pandas dataframe with chrom+start1+start2+length.
    """
    for index, row in signal_table.iterrows():
        signal = row['Signal']
        Format = row['Format']
        Path = row['Path']
        if signal == 'Motif':
            sys.stderr.write("\tAdding motif annotations...\n")
            data = add_motif_pattern(data, Path, opt)        
        elif signal == 'PhastCon':
            sys.stderr.write("\tAdding PhastCons conservation...\n")
            data = add_bigWig_feature(data, Path, opt)
        elif signal == 'Gene expression':
            sys.stderr.write("\tAdding gene expression...\n")
            data = add_gene_expression(data, Path, opt)
        elif signal == "RNA-seq":
            sys.stderr.write("\tAdding gene expression...\n")
            data = add_gene_expr(signal, data, Path, opt)
        else:
            sys.stderr.write("\tProcessing {} features...\n".format(signal))
            if Format == 'bed':
                data = add_features(data, summits, read_info, read_numbers, signal, opt)
            elif Format == 'narrowPeak':
                sys.stderr.write("\t\tPreparing summit signals...\n")
                data = add_peak_feature(signal, data, Path, opt)
                if opt.in_between:
                    # Add "in-between" peaks signal
                    sys.stderr.write("\t\tPreparing in-betweem signals...\n")
                    data = add_peak_inbetween(signal, data, Path, opt)
                if opt.flanking:
                    # Add "upstream" and "downstream" peak signals
                    sys.stderr.write("\t\tPreparing flanking signals...\n")
                    data = add_peak_flanking(signal, data, Path, summits, opt)
    return data

def load_signals_table(info_table):
    """
    Read the signals table from disk.
    """
    signal_table = pd.read_table(info_table)
    signals = []
    for index, row in signal_table.iterrows():
        if (row['Signal'] != 'Motif' and row['Signal'] != 'Gene expression' and row['Signal'] != 'PhastCon'):
            signal = row['Signal']
            signals.append(signal)
    return signal_table, signals


"""
End Added by AGD

"""

def prepare_reads_info(signal_table):
    """
    This function is to prepare the reads info from raw .bed files for the local features.
    Returned: read_info = {factor:{chrom:[start1, start2, start3]}}
    """
    read_info = {}  # read_info = {factor:{chrom:[start1, start2, start3]}}
    read_numbers = {} # read_numbers = {'H3K4me1':read_number, ...}
    shift = 75   # half of the fragment size
    for index, row in signal_table.iterrows():
        factor = row['Signal']
        BED = row['Format']
        Path = row['Path']
        if (factor == 'Reads'):
            BED_reader = open(Path,'r')
            read_info[factor] = {}
            read_number = 0
            for line in BED_reader:
                read_number += 1
                pline = line.strip()
                sline = pline.split('\t')
                chrom = sline[0]
                start = int(sline[1])
                end = int(sline[2])
                strand = sline[5]
                if chrom not in read_info[factor].keys():
                    read_info[factor][chrom] = []

                if (strand == '+'):
                    start = start + shift
                else:
                    start = end - shift

                read_info[factor][chrom].append(start)
            read_numbers[factor] = read_number
            for chrom in read_info[factor].keys():
                read_info[factor][chrom] = sorted(read_info[factor][chrom])
            BED_reader.close()

    return (read_info, read_numbers)


def assign_motif_pattern(strand1, strand2):
    if (strand1 == '-' and strand2 == '+'):
        return '2'
    elif (strand1 == '+' and strand2 == '-'):
        return '4'
    else:
        return '3'


def add_anchor_conservation(train, chrom, Peak):
    """
    To add the feature of sequence conservation on anchors.
    Peak is the folder that contains the phastCon.wig files of all chroms.
    The name format of PhastCon of each chrom is chrXX.phastCons100way.wigFix
    """
    starts = {}
    cvg = {}

    ext = 20
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
    for index, row in train.iterrows():
        con1 = sum(cvg[(int(row['peak1'])-ext): (int(row['peak1'])+ext)])
        con2 = sum(cvg[(int(row['peak2'])-ext): (int(row['peak2'])+ext)])
        AvgCons.append((con1+con2)/2.0)
        DevCons.append(np.std([con1, con2]))
    train['avg_conservation'] = pd.Series(AvgCons)
    train['std_conservation'] = pd.Series(DevCons)

    return train


def add_features(data, anchors, read_info, read_numbers, signals):
    """
    This function is to add both the local and inbetween features to the data.
    read_info = {factor:{chrom:[peak1, peak2, peak3]}}
    inbetween_signals = {factor:{'ChrXX':{summit:peak_height}}}
    anchors = {'chrXX':[peak1, peak2...]}
    """
    extension = 2000

    for factor in signals:
        print "Preparing features for "+str(factor)+'...'
        avg_signal = 'avg_'+str(factor)
        std_signal = 'std_'+str(factor)
        inbetween_signal = str(factor)+'_in-between'
        upstream_signal = str(factor)+'_upstream'
        downstream_signal = str(factor)+'_downstream'

        reads = read_info[factor]
        read_number = read_numbers[factor]
        anchor1_RPKM = []
        anchor2_RPKM = []

        in_between = []
        upstream = []
        downstream = []

        for index, row in data.iterrows():
            chrom = row['chrom']
            peak1 = int(row['peak1'])
            peak2 = int(row['peak2'])

            # Get the RPKM read counts on anchors
            count1 = bisect.bisect_right(reads[chrom], peak1+extension) - bisect.bisect_left(reads[chrom], peak1-extension)
            count2 = bisect.bisect_right(reads[chrom], peak2+extension) - bisect.bisect_left(reads[chrom], peak2-extension)
            RPKM1 = float(count1)/(float(read_number)*2*extension)*1000000000
            RPKM2 = float(count2)/(float(read_number)*2*extension)*1000000000
            anchor1_RPKM.append(np.mean([RPKM1, RPKM2]))
            anchor2_RPKM.append(np.std([RPKM1, RPKM2]))

            # Get the RPKM values of the looped regions
            strength = 0
            count = bisect.bisect_right(reads[chrom], peak2) - bisect.bisect_left(reads[chrom], peak1)
            strength = float(count)/float(abs(peak2 - peak1)*read_number)*1e+9
            in_between.append(strength)
            
            index1 = anchors[chrom].index(peak1)
            index2 = anchors[chrom].index(peak2)
            if index1 != '0':
                up_motif = anchors[chrom][index1 - 1]
                up_count = bisect.bisect_right(reads[chrom],peak1) - bisect.bisect_left(reads[chrom], up_motif)
                up_strength = float(up_count)/float(abs(up_motif-peak1)*read_number)*1e+9
            else:
                up_strength = 0
            upstream.append(up_strength)
            if index2 != (len(anchors[chrom])-1):
                down_motif = anchors[chrom][index2 + 1]
                down_count = bisect.bisect_right(reads[chrom], down_motif) - bisect.bisect_left(reads[chrom], peak2)
                down_strength = float(down_count)/float(abs(down_motif-peak2)*read_number)*1e+9
            else:
                down_strength = 0
            downstream.append(down_strength)

        data[avg_signal] = pd.Series(anchor1_RPKM, index = data.index)
        data[std_signal] = pd.Series(anchor2_RPKM, index = data.index)
        data[inbetween_signal] = pd.Series(in_between, index = data.index)
        data[upstream_signal] = pd.Series(upstream, index = data.index)
        data[downstream_signal] = pd.Series(downstream, index = data.index)
    return data

def add_gene_expression(data, Peak):
    """
    This function is to add the gene expression value of the looped region as a feature.The gene expression file's format is:
    gene_id   locus   value
    A1BG    chr19:coordinate1-coordiate2   1.31
    """
    sys.stderr.write('Preparing features for gene expression...\n')
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
    for index, row in data.iterrows():
        chrom = row['chrom']
        start1 = row['peak1']
        start2 = row['peak2']
        iv = HTSeq.GenomicInterval(chrom, start1, start2)
        loop_expression = 0
        for gene in gene_exp[chrom].keys():
            if gene.overlaps(iv):
                loop_expression += gene_exp[chrom][gene]
        loop_expressions.append(loop_expression)
    data['expression'] = pd.Series(loop_expressions, index = data.index)
    return data
