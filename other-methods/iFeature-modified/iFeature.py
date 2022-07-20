#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import re
import numpy as np
import pandas as pd
from codes import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage="it's usage tip.",
									 description="Generating various numerical representation schemes for protein sequences")
	parser.add_argument("--file", required=True, help="input fasta file")
	parser.add_argument("--type", required=True,
						choices=['All', 'CKSAAP', 'DDE',
								 'GAAC', 'CKSAAGP', 'GDPC', 'GTPC',
								 'CTDC', 'CTDT', 'CTDD',
								 'CTriad', 'KSCTriad'],
						help="the encoding type")
	parser.add_argument("--path", dest='filePath',
						help="data file path used for 'PSSM', 'SSEB(C)', 'Disorder(BC)', 'ASA' and 'TA' encodings")
	parser.add_argument("--train", dest='trainFile',
						help="training file in fasta format only used for 'KNNprotein' or 'KNNpeptide' encodings")
	parser.add_argument("--label", dest='labelFile',
						help="sample label file only used for 'KNNprotein' or 'KNNpeptide' encodings")
	parser.add_argument("--order", dest='order',
						choices=['alphabetically', 'polarity', 'sideChainVolume', 'userDefined'],
						help="output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descriptors")
	parser.add_argument("--userDefinedOrder", dest='userDefinedOrder',
						help="user defined output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descriptors")
	parser.add_argument("--out", dest='outFile',
						help="the generated descriptor file")
	args = parser.parse_args()
	fastas = readFasta.readFasta(args.file)
	userDefinedOrder = args.userDefinedOrder if args.userDefinedOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
	userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
	if len(userDefinedOrder) != 20:
		userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
	myAAorder = {
		'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
		'polarity': 'DENKRQHSGTAPYVMCWIFL',
		'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
		'userDefined': userDefinedOrder
	}
	myOrder = myAAorder[args.order] if args.order != None else 'ACDEFGHIKLMNPQRSTVWY'
	kw = {'path': args.filePath, 'train': args.trainFile, 'label': args.labelFile, 'order': myOrder}
	label = str(args.labelFile)
	output = str(args.outFile)

	# myFun = args.type + '.' + args.type + '(fastas, **kw)'
	desc_cksaap = eval('CKSAAP' + '.' + 'CKSAAP' + '(fastas, **kw)')
	desc_cksaap = pd.DataFrame(desc_cksaap[1:], columns=desc_cksaap[0])
	# print(desc_cksaap)

	desc_dde = eval('DDE' + '.' + 'DDE' + '(fastas, **kw)')
	desc_dde = pd.DataFrame(desc_dde[1:], columns=desc_dde[0])
	# print(desc_dde)

	desc_gaac = eval('GAAC' + '.' + 'GAAC' + '(fastas, **kw)')
	desc_gaac = pd.DataFrame(desc_gaac[1:], columns=desc_gaac[0])
	# print(desc_gaac)

	desc_cksaagp = eval('CKSAAGP' + '.' + 'CKSAAGP' + '(fastas, **kw)')
	desc_cksaagp = pd.DataFrame(desc_cksaagp[1:], columns=desc_cksaagp[0])
	# print(desc_cksaagp)

	desc_gdpc = eval('GDPC' + '.' + 'GDPC' + '(fastas, **kw)')
	desc_gdpc = pd.DataFrame(desc_gdpc[1:], columns=desc_gdpc[0])
	# print(desc_gdpc)

	desc_gtpc = eval('GTPC' + '.' + 'GTPC' + '(fastas, **kw)')
	desc_gtpc = pd.DataFrame(desc_gtpc[1:], columns=desc_gtpc[0])
	# print(desc_gtpc)

	desc_ctdc = eval('CTDC' + '.' + 'CTDC' + '(fastas, **kw)')
	desc_ctdc = pd.DataFrame(desc_ctdc[1:], columns=desc_ctdc[0])
	# print(desc_ctdc)

	desc_ctdt = eval('CTDT' + '.' + 'CTDT' + '(fastas, **kw)')
	desc_ctdt = pd.DataFrame(desc_ctdt[1:], columns=desc_ctdt[0])
	# print(desc_ctdt)

	desc_ctdd = eval('CTDD' + '.' + 'CTDD' + '(fastas, **kw)')
	desc_ctdd = pd.DataFrame(desc_ctdd[1:], columns=desc_ctdd[0])
	# print(desc_ctdd)

	desc_ctraid = eval('CTriad' + '.' + 'CTriad' + '(fastas, **kw)')
	desc_ctraid = pd.DataFrame(desc_ctraid[1:], columns=desc_ctraid[0])
	# print(desc_ctraid)

	desc_ksctriad = eval('KSCTriad' + '.' + 'KSCTriad' + '(fastas, **kw)')
	desc_ksctriad = pd.DataFrame(desc_ksctriad[1:], columns=desc_ksctriad[0])
	# print(desc_ksctriad.iloc[:, 1:])

	df = pd.concat([desc_cksaap.iloc[:, 0:], desc_dde.iloc[:, 1:], desc_gaac.iloc[:, 1:], 
		desc_cksaagp.iloc[:, 1:], desc_gdpc.iloc[:, 1:], desc_gtpc.iloc[:, 1:], desc_ctdc.iloc[:, 1:],
		desc_ctdt.iloc[:, 1:], desc_ctdd.iloc[:, 1:], desc_ctraid.iloc[:, 1:], desc_ksctriad.iloc[:, 1:]], 
		axis=1, ignore_index=False)
	df.rename(columns={"#": "nameseq"}, inplace=True)
	df.insert(len(df.columns), "label", label)


	df.to_csv(output, index=False, mode='a')
	# print(output)
	# print(df)
	# print(label)
	# desc_cksaap.to_csv('test.csv', index=False)
	# outFile = args.outFile if args.outFile != None else 'encoding.csv'
	# saveCode.savetsv(encodings, outFile)
	# python iFeature/iFeature.py --file 1-pvp/TSpos63.fasta --type All --label test --out test2.csv
