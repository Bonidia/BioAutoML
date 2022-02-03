#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import numpy as np
from Bio import SeqIO
import warnings
import sys
import os
import io
from datetime import timedelta, datetime
warnings.filterwarnings("ignore")


#############################################################################
#############################################################################


def sequence_length(finput):
    length = 0
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        if len(seq) > length:
            length = len(seq)
    return length


def file_record(foutput, name_seq, mapping, label_dataset):
    dataset = open(foutput, 'a')
    dataset.write('%s,' % (str(name_seq)))
    for map in mapping:
        dataset.write('%s,' % map)
        # dataset.write('{0:.4f},'.format(metric))
    dataset.write(label_dataset)
    dataset.write('\n')
    # print('Recorded Sequence: %s' % (name_seq))
    return


def binary_mapping(finput, label_dataset, padd):

    var = (datetime.now() + timedelta(hours=9)).strftime('%H%M%S')
    foutput = 'tmp/file-' + var + '.csv'
    if os.path.exists(foutput):
        os.remove(foutput)

    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        A = []
        C = []
        T = []
        G = []
        for nucle in seq:
            if nucle == "A":
                A.append(1)
            else:
                A.append(0)
            if nucle == "C":
                C.append(1)
            else:
                C.append(0)
            if nucle == "T" or nucle == "U":
                T.append(1)
            else:
                T.append(0)
            if nucle == "G":
                G.append(1)
            else:
                G.append(0)
        mapping = A + C + T + G
        if padd == 'Yes':
            padding = (max_length - len(A)) * 4
            mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput


def zcurve_mapping(finput, label_dataset, padd):

    var = (datetime.now() + timedelta(hours=9)).strftime('%H%M%S')
    foutput = 'tmp/file-' + var + '.csv'
    if os.path.exists(foutput):
        os.remove(foutput)

    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        ###################################
        ###################################
        R = 0  # x[n] = (An + Gn) − (Cn + Tn) ≡ Rn − Yn
        Y = 0
        M = 0  # y[n] = (An + Cn) − (Gn + Tn) ≡ Mn − Kn
        K = 0
        W = 0  # z[n] = (An + Tn) − (Cn + Gn) ≡ Wn − Sn
        S = 0
        ###################################
        ###################################
        x = []
        y = []
        z = []
        for nucle in seq:
            if nucle == "A" or nucle == "G":
                R += 1
                x.append(R - Y)
            else:
                Y += 1
                x.append(R - Y)
            if nucle == "A" or nucle == "C":
                M += 1
                y.append(M - K)
            else:
                K += 1
                y.append(M - K)
            if nucle == "A" or nucle == "T" or nucle == "U":
                W += 1
                z.append(W - S)
            else:
                S += 1
                z.append(W - S)
        mapping = x + y + z
        if padd == 'Yes':
            padding = (max_length - len(x)) * 3
            mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput


def integer_mapping(finput, label_dataset, padd):

    var = (datetime.now() + timedelta(hours=9)).strftime('%H%M%S')
    foutput = 'tmp/file-' + var + '.csv'
    if os.path.exists(foutput):
        os.remove(foutput)

    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        mapping = []
        for nucle in seq:
            if nucle == "T" or nucle == "U":
                mapping.append(0)
            elif nucle == "C":
                mapping.append(1)
            elif nucle == "A":
                mapping.append(2)
            else:
                mapping.append(3)
        if padd == 'Yes':
            padding = (max_length - len(mapping))
            mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput


def real_mapping(finput, label_dataset, padd):

    var = (datetime.now() + timedelta(hours=9)).strftime('%H%M%S')
    foutput = 'tmp/file-' + var + '.csv'
    if os.path.exists(foutput):
        os.remove(foutput)

    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        mapping = []
        for nucle in seq:
            if nucle == "T" or nucle == "U":
                mapping.append(1.5)
            elif nucle == "C":
                mapping.append(0.5)
            elif nucle == "A":
                mapping.append(-1.5)
            else:
                mapping.append(-0.5)
        if padd == 'Yes':
            padding = (max_length - len(mapping))
            mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput


def eiip_mapping(finput, label_dataset, padd, foutput):
    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(finput, "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        mapping = []
        for nucle in seq:
            if nucle == "T" or nucle == "U":
                mapping.append(0.1335)
            elif nucle == "C":
                mapping.append(0.1340)
            elif nucle == "A":
                mapping.append(0.1260)
            else:
                mapping.append(0.0806)
        if padd == 'Yes':
            padding = (max_length - len(mapping))
            mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput


def complex_number(finput, label_dataset, padd):

    var = (datetime.now() + timedelta(hours=9)).strftime('%H%M%S')
    foutput = 'tmp/file-' + var + '.csv'
    if os.path.exists(foutput):
        os.remove(foutput)

    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        mapping = []
        for nucle in seq:
            if nucle == "T" or nucle == "U":
                mapping.append(1-1j)
            elif nucle == "C":
                mapping.append(-1+1j)
            elif nucle == "A":
                mapping.append(1+1j)
            else:
                mapping.append(-1-1j)
        if padd == 'Yes':
            padding = (max_length - len(mapping))
            mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput


def atomic_number(finput, label_dataset, padd):

    var = (datetime.now() + timedelta(hours=9)).strftime('%H%M%S')
    foutput = 'tmp/file-' + var + '.csv'
    if os.path.exists(foutput):
        os.remove(foutput)

    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        mapping = []
        for nucle in seq:
            if nucle == "T" or nucle == "U":
                mapping.append(66)
            elif nucle == "C":
                mapping.append(58)
            elif nucle == "A":
                mapping.append(70)
            else:
                mapping.append(78)
        if padd == 'Yes':
            padding = (max_length - len(mapping))
            mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput
#############################################################################
