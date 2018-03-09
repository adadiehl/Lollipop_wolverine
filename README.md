# Lollipop Wolverine

Lollipop is a machine-learning-based framework for predicting the CTCF-mediated interactome by integrating genetic, epigenetic and gene expression data. In our paper *Predicting CTCF-mediated chromatin interactions by integrating genomic and epigenomic features*（`https://www.biorxiv.org/content/early/2017/12/01/215871`）, it was used for:

* Creating positive and negative training data.
* Training a model that distinguishes positive loops from negative loops.
* Applying the trained model to a cell-type of interest to make *de novo* predictions of CTCF-mediated loops. 

The Wolverine branch was created by Adam Diehl in January 2018, based on the original (at https://github.com/ykai16/Lollipop), and adds the capability to use bed narrowPeak ChIP-seq data sources (signal density used as score), incorporates multiprocessing where possible, and increases configurability through addition of various command-line options.

### Dependencies
Lollipop requires the following packages:


* Numpy `http://www.numpy.org`
* Pandas `http://pandas.pydata.org`
* Scikit-learn `http://scikit-learn.org/stable/`
* HTSeq `https://htseq.readthedocs.io/en/release_0.9.1/`
* multiprocessing
* pyBigWig `https://github.com/deeptools/pyBigWig`
* pytabix `https://pypi.python.org/pypi/pytabix`

We recommend to use [Anaconda python distribution](https://www.anaconda.com/what-is-anaconda/) for installation of the above packages.


### Input Data

For a summary of data used in the original publication, please see Supplementary Methods and Table 1 in the paper.

Lollipop_wolverine adds the ability to use peak-based signals from ChIP-seq and similar experiments, and stores these in Tabix-indexed compressed files for speed and flexibility. PhastCons conservation data are now read in bigWig format, and motifs are read in Tabix-indexed compressed files, for the same reasons. See data/data_table.txt for an example of how to supply the locations and formats of these files (this is changed from the original version).


### Step 1: Preparing positive and negative loops for training purpose.

Usage:

`prepare_training_interactions.py [options]`

Options:
  `-h, --help            `show this help message and exit
  `-p <file>, --peak=<file>
                        `the CTCF peaks or summits in BED format
  `-a <file>, --chiapet=<file>
                        `the interaction file from ChIA-PET experiment
  `-c <file>, --hic=<file>
                        `the CTCF interactions identified by hic data
  `-o <file>, --train=<file>
                        `the resulting file with positive and sampled negative
                        `interactions for training
  `-l MIN_LOOP_SIZE, --min_loop_size=MIN_LOOP_SIZE
                        `Minimum loop size. Default 10,000.
  `-u MAX_LOOP_SIZE, --max_loop_size=MAX_LOOP_SIZE
                        `Maximum loop size. Default 1,000,000.
  -r RATIO, --ratio=RATIO
                        `Ratio of negative to positive interactions. Default 5.
                        `Set to 0 to retain all negative interactions.
  -z, --use_hic         `Use Hi-C data as a supplement to ChIA-pet loops in
                        `finding negative loops.
																															  
### Step 2: Characterizing prepared loops.

Usage: add_features.py [options]

Options:
  -h, --help            show this help message and exit
  -i <file>, --training=<file>
                        the training interactions generated in previous step
  -t <file>, --table=<file>
                        the infomation table contains the paths for necessary
                        files.
  -o <file>, --outfile=<file>
                        the output file with all the interactions and
                        calculated features.
  -p PROCS, --procs=PROCS
                        Number of processors to use. Default 1.
  -e EXTENSION, --extension=EXTENSION
                        Size of the extesion window to append up and
                        downstream of each anchor peak for signal calculation.
                        Default 2000. Set to 0 to use actual element
                        boundaries.
  -m MOTIF_EXTENSION, --motif_extension=MOTIF_EXTENSION
                        Size of extension window to append when finding motif
                        features. Default 500. Set to 0 to use actual element
                        boundaries.
  -z CONS_EXTENSION, --cons_extension=CONS_EXTENSION
                        Size of extension window to append when calculating
                        conservation. Default 20. Set to 0 to use actual
                        element boundaries.
  -c COLLAPSE_PEAKS, --collapse_peaks=COLLAPSE_PEAKS
                        How to handle multiple overlapping peak features.
                        Allowed values: max (use maximum score), avg (average
                        all scores), min (use lowest score), sum (use sum of
                        all scores). Default = max.
  -n, --no_in_between_peaks
                        Do not include "in-between" peaks in training and
                        predictions. Our experimentation shows these may
			negatively impact performance.
  -g, --no_flanking_peaks
                        Do not include "upstream" and "downstream" peaks in
                        training and predictions. Our experimentation shows
			these may negatively impact performance.


### Step 3: Model training

A model can be generated from the prepared training data, by using `train_model.py`.

Usage: train_models.py [options]

Options:
  -h, --help            show this help message and exit
  -t <file>, --train=<file>
                        The path of the training data
  -o <file>, --output=<file>
                        The complete path for the resulting model and relevant
                        results
  -p PROCS, --procs=PROCS
                        Number of processors to use. Default=1.
  -n N_ESTIMATORS, --n_estimators=N_ESTIMATORS
                        The number of trees in the random forest. Default 100.
  -m MAX_FEATURES, --max_features=MAX_FEATURES
                        Maximum number of features. Default 18. Specify -1 for
                        all features.


### Step 4: Making *De Novo* Predictions

Lollipop employs a random forest classifier to distinguish positive from negative loops. The classifier trained from the three cell-lines (in `.pkl` format) and the *de novo* predictions made by each classicier are available in `denovo_predictions`. Results are reported in standard bedpe format, with the reported loop probability given in the "score" column.


Usage: make_denovo_predictions.py [options]

Options:
  -h, --help            show this help message and exit
  -b <file>, --bs=<file>
                        the CTCF ChIP-Seq peak file
  -t <file>, --table=<file>
                        the infomation table contains the paths of necessary
                        files.
  -c <file>, --clf=<file>
                        The trained classifier for identifying the interacting
                        loop pairs
  -o OUTDIR, --outdir=OUTDIR
                        The directory for output files, predicted loops, etc
  -p PROCS, --procs=PROCS
                        Number of processors to use. Default 1.
  -u PROXIMAL, --proximal=PROXIMAL
                        Minimum distance between upstream and downstream loop
                        anchors. Default 10,000.
  -d DISTAL, --distal=DISTAL
                        Maximum distance between upstream and downstream loop
                        anchors. Default 1e+6.
  -e EXTENSION, --extension=EXTENSION
                        Size of the extesion window to append up and
                        downstream of each anchor peak for signal calculation.
                        Default 2000 (as in original). Set to 0 to use actual
                        element boundaries.
  -m MOTIF_EXTENSION, --motif_extension=MOTIF_EXTENSION
                        Size of extension window to append when finding motif
                        features. Default 500. Set to 0 to use actual element
                        boundaries.
  -z CONS_EXTENSION, --cons_extension=CONS_EXTENSION
                        Size of extension window to append when calculating
                        conservation. Default 20. Set to 0 to use actual
                        element boundaries.
  -a COLLAPSE_PEAKS, --collapse_peaks=COLLAPSE_PEAKS
                        How to handle multiple overlapping peak features.
                        Allowed values: max (use maximum score), avg (average
                        all scores), min (use lowest score), sum (use sum of
                        all scores). Default = max.
  -f CTCF_F, --ctcf_f=CTCF_F
                        Tabix-indexed CTCF peaks file in narrowPeak format.
  -r, --report_extension
                        Report actual ChIP-seq peak boundaries in output
                        instead of peak +- extension.
  -n, --no_in_between_peaks
                        Do not include "in-between" peaks in training and
                        predictions.
  -g, --no_flanking_peaks
                        Do not include "upstream" and "downstream" peaks in
                        training and predictions.
