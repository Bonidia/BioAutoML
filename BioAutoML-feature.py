import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import argparse
import subprocess
import shutil
import sys
import os.path
import time
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.preprocessing import LabelEncoder
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/other-methods/')
from ChaosGameTheory import *
from MappingClass import *

# Testing
# python BioAutoML-feature.py -fasta_train Case\ Studies/CS-II/train/miRNA.fasta Case\ Studies/CS-II/train/pre_miRNA.fasta -fasta_label_train miRNA pre_miRNA -fasta_test Case\ Studies/CS-II/test/miRNA.fasta Case\ Studies/CS-II/test/pre_miRNA.fasta -fasta_label_test miRNA pre_miRNA -output results/

def objective_rf(space):

	"""Automated Feature Engineering - Objective Function - Bayesian Optimization"""

	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'Chaos': list(range(464, len(train.columns)))}

	for descriptor, ind in descriptors.items():
		if int(space[descriptor]) == 1:
			index = index + ind

	train = train.iloc[:, index]
	test = test.df.iloc[:, index]

	if int(space['Classifier']) == 0:
		model = RandomForestClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)
	else:
		model = lgb.LGBMClassifier(n_estimators=500, n_jobs=n_cpu)

	if len(ftrain_labels) > 2:
		score = make_scorer(balanced_accuracy_score)
	else:
		score = make_scorer(balanced_accuracy_score)

	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	metric = cross_val_score(model,
							 train,
							 train_labels,
							 cv=kfold,
							 scoring=score,
							 n_jobs=n_cpu).mean()

	return {'loss': -metric, 'status': STATUS_OK}


def feature_engineering():

	"""Automated Feature Engineering - Bayesian Optimization"""

	param = {'NAC': [0, 1], 'DNC': [0, 1],
			 'TNC': [0, 1], 'kGap_di': [0, 1], 'kGap_tri': [0, 1],
			 'ORF': [0, 1], 'Fickett': [0, 1],
			 'Shannon': [0, 1], 'FourierBinary': [0, 1],
			 'FourierComplex': [0, 1], 'Tsallis': [0, 1],
			 'Chaos': [0, 1],
			 'Classifier': [0, 1]}

	space = {'NAC': hp.choice('NAC', [0, 1]),
			 'DNC': hp.choice('DNC', [0, 1]),
			 'TNC': hp.choice('TNC', [0, 1]),
			 'kGap_di': hp.choice('kGap_di', [0, 1]),
			 'kGap_tri': hp.choice('kGap_tri', [0, 1]),
			 'ORF': hp.choice('ORF', [0, 1]),
			 'Fickett': hp.choice('Fickett', [0, 1]),
			 'Shannon': hp.choice('Shannon', [0, 1]),
			 'FourierBinary': hp.choice('FourierBinary', [0, 1]),
			 'FourierComplex': hp.choice('FourierComplex', [0, 1]),
			 'Tsallis': hp.choice('Tsallis', [0, 1]),
			 'Chaos': hp.choice('Chaos', [0, 1]),
			 'Classifier': hp.choice('Classifier', [0, 1])}

	trials = Trials()
	best_tuning = fmin(fn=objective_rf,
				space=space,
				algo=tpe.suggest,
				max_evals=250,
				trials=trials)

	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464)), 'Chaos': list(range(464, len(train.columns)))}

	for descriptor, ind in descriptors.items():
		result = param[descriptor][best_tuning[descriptor]]
		if result == 1:
			index = index + ind

	best_train = train.iloc[:, index]
	best_test = test.df.iloc[:, index]

	return best_train, best_test


def feature_extraction(ftrain, ftrain_labels, ftest, ftest_labels, features, foutput):

	"""Extracts the features from the sequences in the fasta files."""

	path = 'feat_extraction'
	path_results = foutput

	print('Extracting features with MathFeature...')
	start_time = time.time()

	try:
		shutil.rmtree(path)
		shutil.rmtree(path_results)
	except OSError as e:
		print("Error: %s - %s." % (e.filename, e.strerror))

	if not os.path.exists(path) or not os.path.exists(path_results):
		os.mkdir(path)
		os.mkdir(path + '/train')
		os.mkdir(path + '/test')
		os.mkdir(path_results)

	labels = [ftrain_labels, ftest_labels]
	fasta = [ftrain, ftest]
	train_size = 0

	datasets = []

	for i in range(2):
		for j in range(len(labels[i])):
			file = fasta[i][j].split('/')[-1]
			if i == 0: # Train
				preprocessed_fasta = path + '/train/pre_' + file
				subprocess.call(['python', 'MathFeature/preprocessing/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				train_size += len([1 for line in open(preprocessed_fasta) if line.startswith(">")])
			else: # Test
				preprocessed_fasta = path + '/test/pre_' + file
				subprocess.call(['python', 'MathFeature/preprocessing/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

			if 1 in features:
				dataset = path + '/NAC.csv'
				subprocess.call(['python', 'MathFeature/methods/ExtractionTechniques.py',
								'-i', preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'NAC', '-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 2 in features:
				dataset = path + '/DNC.csv'
				subprocess.call(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'DNC', '-seq', '1'], stdout=subprocess.DEVNULL,
								stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 3 in features:
				dataset = path + '/TNC.csv'
				subprocess.call(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'TNC', '-seq', '1'], stdout=subprocess.DEVNULL,
								stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 4 in features:
				dataset_di = path + '/kGap_di.csv'
				dataset_tri = path + '/kGap_tri.csv'

				subprocess.call(['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', dataset_di, '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '2', '-seq', '1'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

				subprocess.call(['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', dataset_tri, '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '3', '-seq', '1'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset_di)
				datasets.append(dataset_tri)

			if 5 in features:
				dataset = path + '/ORF.csv'
				subprocess.call(['python', 'MathFeature/methods/CodingClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j]],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 6 in features:
				dataset = path + '/Fickett.csv'
				subprocess.call(['python', 'MathFeature/methods/FickettScore.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 7 in features:
				dataset = path + '/Shannon.csv'
				subprocess.call(['python', 'MathFeature/methods/EntropyClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-k', '5', '-e', 'Shannon'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 8 in features:
				dataset = path + '/FourierBinary.csv'
				subprocess.call(['python', 'MathFeature/methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-r', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 9 in features:
				dataset = path + '/FourierComplex.csv'
				subprocess.call(['python', 'MathFeature/methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-r', '6'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 10 in features:
				dataset = path + '/Tsallis.csv'
				subprocess.call(['python', 'MathFeature/methods/TsallisEntropy.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-k', '5', '-q', '2.3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 11 in features:
				dataset = path + '/Chaos.csv'
				classifical_chaos(preprocessed_fasta, labels[i][j], 'Yes', dataset)
				datasets.append(dataset)

			if 12 in features:
				dataset = path + '/BinaryMapping.csv'
				binary_mapping(preprocessed_fasta, labels[i][j], 'Yes', dataset)
				datasets.append(dataset)

	"""Concatenating all the extracted features"""

	if datasets:
		datasets = list(dict.fromkeys(datasets))
		dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
		dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
		dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]

	X_train = dataframes.iloc[:train_size,:]
	X_test = dataframes.iloc[train_size:,:]

	y_train = X_train.pop('label')
	y_test = X_test.pop('label')

	nameseq_train = X_train.pop('nameseq')
	nameseq_test = X_test.pop('nameseq')

	fnameseqtrain = path + '/fnameseqtrain.csv'
	nameseq_train.to_csv(fnameseqtrain, index = False, header = True)
	fnameseqtest = path + '/fnameseqtest.csv'
	nameseq_test.to_csv(fnameseqtest, index = False, header = True)
	ftrain = path + '/ftrain.csv'
	X_train.to_csv(ftrain, index=False)
	ftest = path + '/ftest.csv'
	X_test.to_csv(ftest, index=False)
	flabeltrain = path + '/flabeltrain.csv'
	y_train.to_csv(flabeltrain, index = False, header = True)
	flabeltest = path + '/flabeltest.csv'
	y_test.to_csv(flabeltest, index = False, header = True)

	cost = (time.time() - start_time)/60
	print('Computation time - MathFeature: %s minutes' % cost)

	return fnameseqtest, ftrain, ftest, flabeltrain, flabeltest

##########################################################################
##########################################################################
if __name__ == '__main__':
	print('\n')
	print('###################################################################################')
	print('###################################################################################')
	print('##########         BioAutoML- Automated Feature Engineering             ###########')
	print('##########              Author: Robson Parmezan Bonidia                 ###########')
	print('##########         WebPage: https://bonidia.github.io/website/          ###########')
	print('###################################################################################')
	print('###################################################################################')
	print('\n')
	parser = argparse.ArgumentParser()
	parser.add_argument('-fasta_train', '--fasta_train', nargs='+',
						help='fasta format file, e.g., fasta/ncRNA.fasta'
							 'fasta/lncRNA.fasta fasta/circRNA.fasta')
	parser.add_argument('-fasta_label_train', '--fasta_label_train', nargs='+',
						help='labels for fasta files, e.g., ncRNA lncRNA circRNA')
	parser.add_argument('-fasta_test', '--fasta_test', nargs='+',
						help='fasta format file, e.g., fasta/ncRNA fasta/lncRNA fasta/circRNA')
	parser.add_argument('-fasta_label_test', '--fasta_label_test', nargs='+',
						help='labels for fasta files, e.g., ncRNA lncRNA circRNA')
	parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	ftrain = args.fasta_train
	ftrain_labels = args.fasta_label_train
	ftest = args.fasta_test
	ftest_labels = args.fasta_label_test
	n_cpu = int(args.n_cpu)
	foutput = str(args.output)

	for fasta in ftrain:
		if os.path.exists(fasta) is True:
			print('Train - %s: Found File' % fasta)
		else:
			print('Train - %s: File not exists' % fasta)
			sys.exit()

	for fasta in ftest:
		if os.path.exists(fasta) is True:
			print('Test - %s: Found File' % fasta)
		else:
			print('Test - %s: File not exists' % fasta)
			sys.exit()

	features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	nameseq_test, ftrain, ftest, \
		ftrain_labels,ftest_labels = feature_extraction(ftrain, ftrain_labels, ftest, ftest_labels, features, foutput)

	start_time = time.time()

	cost = (time.time() - start_time)/60
	print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
