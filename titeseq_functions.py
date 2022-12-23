# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:04:22 2022

@author: marsh
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial

#sequences of variants - all padded with ' ' at beginning to align mutations with sequence
B1351_seq      =  ' NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST'
E484K_seq      =  ' NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
N5O1Y_seq      =  ' NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST'
Wuhan_Hu_1_seq =  ' NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
WT_seq         =  ' NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
delta_seq      =  ' NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

ids = ['B1351','E484K','Wuhan_Hu_1','N501Y','Delta']
sequences = [B1351_seq,E484K_seq,Wuhan_Hu_1_seq,N5O1Y_seq,delta_seq]
seq_dict = dict(zip(ids,sequences))

def preprocess(values_counts,values,index,columns):
    '''
    preprocess: convert dataframe from the following format:
    index,barcode,pool,replicate,bin #,count
    0,ATACTTATGTATAGAC,pool1,1,1-1,0
    1,ATACTTATGTATAGAC,pool1,2,1-1,16
    2,ATACTTATGTATAGAC,pool1,1,1-2,23
    3,ATACTTATGTATAGAC,pool2,1,1-2,5
    ...
    
    to:
    barcode,pool,replicate,[bins]
    ATACTTATGTATAGAC,pool1,1,0,23,...
    ATACTTATGTATAGAC,pool1,2,16,...
    ATACTTATGTATAGAC,pool2,1,0,5,...
    ...
    
    Inputs:
        values_counts: dataframe of identifying information, bins, and NGS read counts
        values: column of NGS read counts
        index: list like of identifying information
        columns: column of bin identifiers
    Outputs:
        bin_counts: dataframe broken down by index information and # of reads in bins
    '''
    return pd.pivot_table(values_counts,values=values,index=index,columns=columns)

def getFrequencies(bin_counts):
    '''
    getFrequencies: transform the data from NGS read counts to frequencies
    Inputs:
        bin_counts: dataframe broken down by index information and # of reads in bins
    Outputs:
        bin_frequencies: dataframe broken down by index information and frequency of reads in bins
    '''
    bin_counts = bin_counts.fillna(0)
    bin_frequencies = bin_counts / bin_counts.sum(axis=0)
    return bin_frequencies

def filterClones(bin_frequencies,bin_read_fraction=0.5,bin_read_threshold=1,bimodal_removal=True,bimodal_removal_threshold=0.4,
                 minimum_fraction_removal=0,num_bins=4,num_concs=10):
    '''
    filterClones: filter clones based on quality metrics
    Inputs:
        bin_frequencies: dataframe broken down by index information and frequency of reads in bins
        bin_read_fraction: 0<=x <= 1, remove clones that appear in less than x bins at bin_read_threshold specified concentrations (default 0.5)
        bin_read_threshold: 0<=x <= 1, determines what fraction of concs need to appear for bin_read_fraction (default 1)
        bimodal_removal: boolean, whether to remove bimodal clones (default True)
        bimodal_removal_threshold: 0<=x <= 1, what fraction of population needs to be bimodal for cut-off (default 0.4)
        minimum_fraction_removal: float, minimum fraction specified for removal (default 0)
        num_bins: int, number of bins (default 4)
        num_concs: int, number of concentrations (default 10)
    Outputs:
        filtered_bin_frequencies: all clones that passed criterion bin_read_threshold
        masked_concs: masked dataframe, 0 if datapoint can be used for fitting, 1 otherwise
    '''
    #array of masked concentrations
    masked_concs = pd.DataFrame(index=bin_frequencies.index,columns=range(num_concs))
    masked_concs = masked_concs.fillna(0)
    
    #filter based on minimum fraction
    bin_frequencies[bin_frequencies <= minimum_fraction_removal] = 0
    
    #filter based on number of bins at each concentration threshold
    conc_sum = pd.Series(index=bin_frequencies.index)
    conc_sum = conc_sum.fillna(0)
    for conc in range(num_concs):
        columns = bin_frequencies.columns[conc*num_bins:conc*num_bins+num_bins]
        conc_sum += ((bin_frequencies[columns] != 0).sum(axis=1)/num_bins >= bin_read_fraction)
        masked_concs.loc[(bin_frequencies[columns] != 0).sum(axis=1)/num_bins < bin_read_fraction,conc] = 1
        
    #filter based on number of concentrations threshold (bin_read_threshold)
    filtered_bin_frequencies = bin_frequencies.loc[conc_sum.values/ num_concs >= bin_read_threshold]
    
    #filter bimodal populations
    if bimodal_removal:
        for conc in range(num_concs):
            columns = bin_frequencies.columns[conc*num_bins:conc*num_bins+num_bins]
            bimodal_columns = [[columns[i],columns[j]] for i in range(len(columns)) for j in range(len(columns)) if j - i >= 2]
            for c in bimodal_columns:
                # masked_concs[(bin_frequencies.loc[:,c].sum(axis=1) >= (bin_frequencies.loc[:,columns].sum(axis=1)*bimodal_removal_threshold))] = 1
                
                masked_concs.loc[(bin_frequencies.loc[:,c[0]] >= bin_frequencies.loc[:,columns].sum(axis=1)*bimodal_removal_threshold) & 
                             (bin_frequencies.loc[:,c[1]] >= bin_frequencies.loc[:,columns].sum(axis=1)*bimodal_removal_threshold) & 
                             (bin_frequencies.loc[:,columns].sum(axis=1)*bimodal_removal_threshold > 0),conc] = 1
    
    
    # return bin_frequencies,masked_concs
    return filtered_bin_frequencies,masked_concs

def getBinScores(bin_frequencies,masked_concs,num_concs=10,num_bins=4):
    '''
    getBinScores: convert dataframe of bin frequencies and masked concentration points
    to fit-ready data points

    Parameters
    ----------
    bin_frequencies : dataframe
        from getFrequencies()
    masked_concs : dataframe
        from filterClones()
    num_concs : int
        number of concentrations (default 10)
    num_bins : int
        number of bins (default 4)
        
    Returns
    -------
    fit_data: fit-ready data points (np.nan are to be excluded)
    
    '''
    
    fit_data = pd.DataFrame(index=bin_frequencies.index,columns=range(num_concs))
    for conc in range(num_concs):
        columns = bin_frequencies.columns[conc*num_bins:conc*num_bins+num_bins]
        fit_data.loc[:,conc] = bin_frequencies[columns].multiply(np.arange(1,num_bins+1),axis=1).sum(axis=1) / bin_frequencies[columns].sum(axis=1)
    
    fit_data[masked_concs == 1] = np.nan
    return fit_data

def sigmoid(x, bottom,top,Kd):
    '''
    sigmoid: standard form of sigmoid for curve fitting
    
    Parameters
    ----------
    x : array
        concentration vector [M]
    bottom : float
        value of y at low x
    top : float
         value of y at high x
    Kd : float
        concentration at half-maximum output

    Returns
    -------
    y: array
        sigmoid of inputs according to (L / (1 + np.exp(-k*(x-x0))) + b)

    '''
    return (bottom + x*(top-bottom)/(Kd+x))

def plotData(fit_data,barcode,concs,num_bins=4,ax=None):
    '''
    plotData: plot a barcode's titration

    Parameters
    ----------
    fit_data : dataframe
        fit-ready data points (np.nan are to be excluded), from getBinScores()
    barcode : tuple
        identifier of molecule (DNA: str, pool: str, replicate: str)
    concs: array
        concentration vector
    num_bins: int
        number of bins, default 4
    ax: axes object
        axes object to plot on, default None (new plot)

    Returns
    -------
    ax: axes
        axes object of plot

    '''
    if ax is None:
        _,ax = plt.subplots()

    x = concs[1:]
    y = fit_data.loc[barcode].values[1:]
    x_domain = [0.1*np.ma.masked_equal(x, 0.0, copy=False).min(),10*max(x)]
    ax.scatter(x,y,color='black')
    ax.set_xscale('log')
    ax.set_xlim(x_domain)
    ax.set_ylim([0,num_bins+1])
    ax.plot(x_domain,[1,1],linestyle='dashed')
    ax.plot(x_domain,[num_bins,num_bins],linestyle='dashed')
    return ax

def curveFit(fit_data,barcode,concs,num_bins=4,verbose=False):
    '''
    
    Parameters
    ----------
    fit_data : dataframe
        fit-ready data points (np.nan are to be excluded), from getBinScores()
    barcode : tuple
        identifier of molecule (DNA: str, pool: str, replicate: str)
    concs : array
        concentration vector
    num_bins : int, optional
        number of bins, default 4
    verbose : bool, optional
        if True, print error messages, default False

    Returns
    -------
    popt: array
        parameters from optimization
    pcov: array
        covariances of parameters from optimziation
        
    '''
    x = np.array(concs[1:])
    y = fit_data.loc[barcode].values[1:]
    valid = ~(np.isnan(x) | np.isnan(y))
    if sum(valid) == 0:
        if verbose:
            print('no data for this barcode')
        return np.zeros(3),np.zeros((3,3))
    bounds = ((1, 1, min(x)), (num_bins, num_bins, 10*max(x)))
    p0 = [max(y[valid]), min(y[valid]), np.median(x)]
    try:
        popt, pcov = curve_fit(sigmoid, x[valid], y[valid],p0, bounds=bounds,method='trf')
    except TypeError: #fewer datapoints than parameters
        if verbose:
            print('too few data points for this barcode')
        return np.zeros(3),np.zeros((3,3))
    except ValueError: #all nan
        if verbose:
            print('no data for this barcode')
        return np.zeros(3),np.zeros((3,3))
    except RuntimeError: #not converging
        if verbose:
            print('Optimal parameters not found: The maximum number of function evaluations is exceeded.')
        return np.zeros(3),np.zeros((3,3))
    except OptimizeWarning: #covariance issue
        if verbose:
            print('Covariance of the parameters could not be estimated.')
            return np.zeros(3),np.zeros((3,3))
    if popt[0] < popt[1]:
        return popt,pcov
    else:
        if verbose:
            print('spurious fit')
        return np.zeros(3),np.zeros((3,3))

def plotCurveFit(fit_data,barcode,concs,num_bins=4,ax=None):
    '''
    

    Parameters
    ----------
    fit_data : dataframe
        fit-ready data points (np.nan are to be excluded), from getBinScores()
    barcode : tuple
        identifier of molecule (DNA: str, pool: str, replicate: str)
    concs : array
        concentration vector
    num_bins : int, optional
        number of bins, default 4
    ax: ax
        axes to plot on, default None

    Returns
    -------
    ax: axes
        axes object of plot

    '''
    if ax is None:
        _,ax = plt.subplots()
    ax = plotData(fit_data,barcode,concs,num_bins,ax=ax)
    popt,pcov = curveFit(fit_data,barcode,concs,num_bins)
    x = np.array(concs[1:])
    y = fit_data.loc[barcode].values[1:]
    x_domain = [0.1*np.ma.masked_equal(x, 0.0, copy=False).min(),10*max(x)]
    x_range = np.logspace(np.log10(min(x_domain)),np.log10(max(x_domain)))
    y_range = sigmoid(x_range,*popt)
    ax.plot(x_range,y_range,color='black')
    return ax

def curveFitDataset(fit_data,concs,num_bins=4,**kwargs):
    '''
    curveFitDataset: apply curveFit() over an entire dataset

    Parameters
    ----------
    fit_data : dataframe
        fit-ready data points (np.nan are to be excluded), from getBinScores()
    concs : array
        concentration vector
    num_bins : int, optional
        number of bins, default 4

    Returns
    -------
    popts: dataframe
        all fit parameters

    '''
    popts = pd.DataFrame(index=fit_data.index,columns=['bottom','top','Kd'])
    for barcode in fit_data.index:
        popt, _ = curveFit(fit_data,barcode,concs,**kwargs)
        popts.loc[barcode,['bottom','top','Kd']] = popt
    return popts

def multiprocessFit(fit_data,ncpus,concs,n_bins=4,**kwargs):
    '''
    multiprocessFit: use multiple CPUs to speed up curve fitting

    Parameters
    ----------
    fit_data : dataframe
        fit-ready data points (np.nan are to be excluded), from getBinScores()
    ncpus : int
        number of CPU's to use
    concs : array
        concentration vector
    num_bins : int
        number of bins, default 4
        
    Returns
    -------
    fit_params: parameters from fitting (top, bottom, Kd)

    '''
    pool = Pool(ncpus)
    batch_size = (len(fit_data) // ncpus) + 1
    batches = [fit_data.iloc[i : i + batch_size] for i in range(0, len(fit_data), batch_size)]
    fit_params = pool.map(partial(curveFitDataset,concs=concs,**kwargs),batches)
    fit_params = pd.concat(fit_params)
    return fit_params

def plotDatasetHistogram(fit_params,concs,n_bins=50,ax=None,**kwargs):
    '''
    plotDatasetHistogram: plot distribution of fit Kd's
    
    Parameters
    ----------
    fit_params: dataframe
        parameters from fitting (top, bottom, Kd)
    concs : array
        concentration vector
    n_bins : int
        number of bins for histogram, default 50
    ax: ax
        axes to plot on
    
    Returns
    -------
    ax: ax
        axes plotted on
    '''
    if ax is None:
        _,ax = plt.subplots()
    x = np.array(concs[1:])
    threshold = 0.1*np.ma.masked_equal(x, 0.0, copy=False).min()
    good_fit_params = fit_params.loc[fit_params['Kd'] > threshold]['Kd']
    logbins = np.logspace(np.log10(good_fit_params.min()), np.log10(good_fit_params.max()), 50)
    ax.hist(good_fit_params, bins=logbins)
    ax.set_xscale('log')
    ax.set_ylabel('Count')
    ax.set_xlabel('Affinity [M]')
    return ax
        
def mutate_sequence(background,aa_substitutions,verbose=False):
    '''
    mutate_sequence: given a background protein sequence, generate a new one with mutations in a string like: [original token][position][new token]
    for example 'A155E' or 'A155E S225P'
    
    Parameters
    ----------
    background: string
        name of background for indexing in seq_dict
    aa_substitutions : string
        mutations to make, in a string
    verbose : boolean
        whether to output errors, default False
    
    Returns
    -------
    new_seq: string
        new sequence with mutations incorporated
    
    '''
    try: #catch entry not in dictionary
        new_seq = seq_dict[background]
    except ValueError:
        if verbose:
            print('variant sequence not found in dictionary')
        return ''
    except KeyError:
        if verbose:
            print('variant sequence not found in dictionary')
        return ''
    
    try: #catch np.NaN submitted 
        muts = aa_substitutions.split(' ')
    except AttributeError:
        if verbose:
            print('no mutations')
        return ''
    
    if (len(muts) == 0):
        return ''
    
    for i,m in enumerate(muts): #iterate through the mutations and change the string
        wt = m[0]
        mut = m[-1]
        pos = int(m[1:-1])
        if pos > len(new_seq):
            if verbose:
                print('invalid position, skipping')
            pass
        else:
            new_seq = new_seq[:pos] + mut + new_seq[pos + 1:]
    return new_seq


def getRatios(input_data,data_type):
    '''
    getRatios: given a titeseq dataset, calculate ratios for sequences and return processed data
    
    Parameters
    ----------
    input_data: dataframe
        dataframe with the sequence as the index and two columns (per replicate if illumina), high_R and low_R where R is each replicate lettered from A->Z
    data_type: str
        if 'duplicated', assumes there are duplicate sequences (due to independent barcodes or degenerate codons) (like Bloom 2022 or Kinney 2016)
        if 'nonduplicated', assumes there are no duplicate sequences  
    
    Outputs
    -------
    output_data: dataframe
        dataframe with sequences and ratios, mean, and count statistics calculated
    
    '''
    #create copy to prevent dataframe issues in scripts
    binary_data = input_data.copy()
    try:
        num_replicates = int(len(binary_data.columns) / 2)
    except:
        print('odd number of columns submitted')
        return None
    
    replicates = ['A','B','C','D','E','F'][:num_replicates]
    negative_column = 'low_'
    positive_column = 'high_'
    
    #convert read counts to frequencies and frequencies to ratios
    for r in replicates:
        binary_data[positive_column+r] = binary_data[positive_column+r]/binary_data[positive_column+r].sum(axis=0)
        binary_data[negative_column+r] = binary_data[negative_column+r]/binary_data[negative_column+r].sum(axis=0)
        binary_data['ratio_' + r] = np.where(binary_data[negative_column+r] == 0, binary_data[positive_column+r] / (binary_data[binary_data[negative_column+r] != 0][negative_column+r].min(axis=0)), binary_data[positive_column+r] / binary_data[negative_column+r])
    
    #remove ratios if they're zero
    for r in replicates:
        binary_data.loc[(binary_data['low_' + r] == 0) & (binary_data['high_' + r] == 0),'ratio_'+r] = np.nan
        
    #drop rows where all ratios are nan
    binary_data = binary_data.loc[pd.isnull(binary_data[['ratio_' + r for r in replicates]]).sum(axis=1) < num_replicates]
    
    #if pacbio, group identical sequences by ratios and calculate statistics
    if data_type == 'duplicated':        
        df1 = pd.DataFrame(index=binary_data.index.unique(),columns=binary_data.columns)
        binary_data['ratios'] = binary_data[['ratio_' + r for r in replicates]].values.tolist()
        df1['ratios'] = binary_data.groupby('sequence')['ratios'].apply(lambda x: list([item for sublist in x for item in sublist]))
        df1 = df1.drop(columns=df1.columns[:-1])
        df1['ratios'] = df1['ratios'].apply(lambda x: [i for i in x if not np.isnan(i)])
        df1['mean'] = df1['ratios'].apply(lambda x: np.nanmean(x))
        df1['count'] = df1['ratios'].apply(lambda x: len(x))
        binary_data = df1
    elif data_type == 'nonduplicated':
        binary_data['ratios'] = binary_data[['ratio_' + r for r in replicates]].values.tolist()
        binary_data['ratios'] = binary_data['ratios'].apply(lambda x: [i for i in x if not np.isnan(i)])
        binary_data['mean'] = binary_data['ratios'].apply(lambda x: np.nanmean(x))
        binary_data['count'] = binary_data['ratios'].apply(lambda x: len(x))
        
    return binary_data

def generateBinaryDataset(input_data,data_type,replicate_cutoff,percentile_cutoff,zero_tolerance=False,isolated_replicates=False):
    '''
    generateBinaryDataset: given a titeseq dataset, convert it to a binary dataset based on its sequencing type and various cutoff parameters
    
    Parameters
    ----------
    input_data: dataframe
        dataframe with the sequence as the index and two columns (per replicate if illumina), high_R and low_R where R is each replicate lettered from A->Z
    data_type: str
        if 'duplicated', assumes there are duplicate sequences (due to independent barcodes or degenerate codons) (like Bloom 2022 or Kinney 2016)
        if 'nonduplicated', assumes there are no duplicate sequences
    replicate_cutoff: tuple of (float,float)
        determines how strict to be for determination of positive and negative dataset respectively based on # of replicates
        if data_type = 'nonduplicated', this is based on the number of replicates of the entire dataset since identical sequences are pooled
        if data_type = 'duplicated', this can be any number depending on how many identical sequences were prepared in the barcoding process
    percentile_cutoff: tuple of (float,float)
        determines how strict to be for determination of positive and negative dataset respectively based on the distribution of ratios in the data
    zero_tolerance: bool
        if True, negative dataset only includes sequences with ratio 0 (ignoring percentile cutoff). default False
    isolated_replicates: bool
        if True and data_type is 'nonduplicated', identifies percentile cutoffs based on individual replicates instead of pooling. default False
    
    Outputs
    -------
    positive_labels: dataframe
        dataset of positive sequences
    negative_labels: dataframe
        dataset of negative sequences
    '''
    #create copy to prevent dataframe issues in scripts
    binary_data = input_data.copy()
    #get ratios
    if 'ratios' in binary_data.columns:
        pass
    else:
        binary_data = getRatios(binary_data,data_type)
        
    #apply filters to the data
    if isolated_replicates and data_type == 'nonduplicated':
        positive_labels = binary_data.loc[(binary_data[['ratio_' + r for r in replicates]] >= binary_data[['ratio_' + r for r in replicates]].quantile(percentile_cutoff[0])).sum(axis=1) >= replicate_cutoff[0]]
    else:
        positive_labels = binary_data.loc[(binary_data['mean'] > binary_data['mean'].quantile(percentile_cutoff[0])) & (binary_data['count'] >= replicate_cutoff[0])]
        
    if zero_tolerance:
        negative_labels = binary_data.loc[(binary_data['mean'] == 0) & (binary_data['count'] >= replicate_cutoff[1])]
    else:
        negative_labels = binary_data.loc[(binary_data['mean'] < binary_data['mean'].quantile(percentile_cutoff[1])) & (binary_data['count'] >= replicate_cutoff[1])]
    
    positive_labels['label'] = 1
    negative_labels['label'] = 0
    
    return positive_labels[['mean','label']],negative_labels[['mean','label']]

def plotDatasetCutoff(input_data,data_type,replicate_cutoffs,percentile_cutoffs,zero_tolerance=True):
    '''
    plotDatasetCutoff: plot the size of a simulated binary dataset based on various cutoff parameters
    
    Parameters
    ----------
    input_data: dataframe
        dataframe with the sequence as the index and two columns (per replicate if illumina), high_R and low_R where R is each replicate lettered from A->Z
    data_type: str
        if 'duplicated', assumes there are duplicate sequences (due to independent barcodes or degenerate codons) (like Bloom 2022 or Kinney 2016)
        if 'nonduplicated', assumes there are no duplicate sequences
    replicate_cutoffs: array of floats 0<= x <= 1
        determines how strict to be for determination of positive and negative dataset respectively based on # of replicates
    percentile_cutoffs: array of floats 0<= x <= 1
        determines how strict to be for determination of positive and negative dataset respectively based on the distribution of ratios in the data
    zero_tolerance: bool
        if True, includes the data point that ignores percentile cutoff and only takes negative datapoints with 0 ratio. default True
    
    
    Outputs:
    --------
    None
    
    '''
    #create copy to prevent dataframe issues in scripts
    binary_data = input_data.copy()
    #get ratios
    if 'ratios' in binary_data.columns:
        pass
    else:
        binary_data = getRatios(binary_data,data_type)
    #add zero tolerance to negative dataset if applicable
    if zero_tolerance:
        percentile_cutoffs.append('zero_tolerance')
    
    #plot results
    fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(20,10))
    
    for k,ax in enumerate(np.ravel(axes)):
        datasize_df = pd.DataFrame(index=percentile_cutoffs,columns=replicate_cutoffs)
        #get stats
        for i,n in enumerate(replicate_cutoffs): #iterate through replicate cutoffs
            for j,c in enumerate(percentile_cutoffs): #iterate through percentile cutoffs
                data_slice = binary_data[['mean','count']]
                if k == 0: #negative
                    if c == 'zero_tolerance':
                        data_slice = data_slice.loc[(data_slice['mean'] == 0) & (data_slice['count'] > n)]
                    else:
                        data_slice = data_slice.loc[(data_slice['mean'] < data_slice['mean'].quantile(c)) & (data_slice['count'] > n)]
                    label='negative'
                elif k == 1: #positive
                    if c == 'zero_tolerance':
                        data_slice = []
                    else:
                        data_slice = data_slice.loc[(data_slice['mean'] > data_slice['mean'].quantile(c)) & (data_slice['count'] > n)]
                    label='positive'
                    
                datasize_df.loc[c,n] = len(data_slice)
            
        pos = ax.imshow(datasize_df.values.astype('int'))
        xticks = range(len(datasize_df.columns))
        yticks = range(len(datasize_df.index))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(datasize_df.columns)
        ax.set_yticklabels(datasize_df.index)
        ax.set_xlabel('# Replicate Cutoff')
        ax.set_ylabel('Ratio percentile cutoff')
        cbar = fig.colorbar(pos, ax=ax)
        cbar.set_label('Number of datapoints after filtering')
        for (i, j), z in np.ndenumerate(datasize_df):
            ax.text(j, i, str(z), ha='center', va='center',color='white')
        ax.set_title(f'size of {label} dataset')