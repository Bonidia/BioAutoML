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


def objective_feature_selection(space):

	"""Feature Importance-based Feature selection: Objective Function - Bayesian Optimization"""

	t = space['threshold']

	fs = SelectFromModel(clf, threshold=t)
	fs.fit(train, train_labels)
	fs_train = fs.transform(train)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	bacc = cross_val_score(clf,
						   fs_train,
						   train_labels,
						   cv=kfold,
						   scoring=make_scorer(balanced_accuracy_score),
						   n_jobs=n_cpu).mean()

	return {'loss': -bacc, 'status': STATUS_OK}


def feature_importance_fs_bayesian(model, train, train_labels):

	"""Feature Importance-based Feature selection using Bayesian Optimization"""

	model.fit(train, train_labels)
	importances = set(model.feature_importances_)
	importances.remove(max(importances))
	importances.remove(max(importances))

	space = {'threshold': hp.uniform('threshold', min(importances), max(importances))}

	trials = Trials()
	best_threshold = fmin(fn=objective_feature_selection,
					   space=space,
					   algo=tpe.suggest,
					   max_evals=300,
					   trials=trials)

	return best_threshold['threshold']


def feature_extraction():

	"""Extracts the features from the sequences in the fasta files."""

	path = foutput
	path_results = 'feat_extraction_results'

	print('Extracting features with MathFeature...')

	try:
		shutil.rmtree(path)
		shutil.rmtree(path_results)
	except OSError as e:
		print("Error: %s - %s." % (e.filename, e.strerror))

	if not os.path.exists(path) and not os.path.exists(path_results):
		os.mkdir(path)
		os.mkdir(path_results)

	datasets = []

	for i in range(len(fasta_label)):

		file = fasta[i].split('/')[-1]
		preprocessed_fasta = path + '/pre_' + file
		subprocess.call(['python', 'MathFeature/preprocessing/preprocessing.py',
						 '-i', fasta[i], '-o', preprocessed_fasta],
						stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

		if 1 in features:
			dataset = path + '/NAC.csv'
			subprocess.call(['python', 'MathFeature/methods/ExtractionTechniques.py',
							 '-i', preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-t', 'NAC', '-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 2 in features:
			dataset = path + '/DNC.csv'
			subprocess.call(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-t', 'DNC', '-seq', '1'], stdout=subprocess.DEVNULL,
							stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 3 in features:
			dataset = path + '/TNC.csv'
			subprocess.call(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-t', 'TNC', '-seq', '1'], stdout=subprocess.DEVNULL,
							stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 4 in features:
			dataset = path + '/kGap.csv'
			subprocess.call(['python', 'MathFeature/methods/Kgap.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l',
							 fasta_label[i], '-k', '1', '-bef', '1',
							 '-aft', '2', '-seq', '1'],
							stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 5 in features:
			dataset = path + '/ORF.csv'
			subprocess.call(['python', 'MathFeature/methods/CodingClass.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i]],
							stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 6 in features:
			dataset = path + '/Fickett.csv'
			subprocess.call(['python', 'MathFeature/methods/FickettScore.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 7 in features:
			dataset = path + '/Shannon.csv'
			subprocess.call(['python', 'MathFeature/methods/EntropyClass.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-k', '5', '-e', 'Shannon'],
							stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 8 in features:
			dataset = path + '/FourierBinary.csv'
			subprocess.call(['python', 'MathFeature/methods/FourierClass.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-r', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 9 in features:
			dataset = path + '/FourierComplex.csv'
			subprocess.call(['python', 'MathFeature/methods/FourierClass.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-r', '6'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 10 in features:
			dataset = path + '/Tsallis.csv'
			subprocess.call(['python', 'MathFeature/methods/TsallisEntropy.py', '-i',
							 preprocessed_fasta, '-o', dataset, '-l', fasta_label[i],
							 '-k', '5', '-q', '2.3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			datasets.append(dataset)

		if 11 in features:
			dataset = path + '/EIIP.csv'
			eiip_mapping(preprocessed_fasta, fasta_label[i], 'Yes', dataset)
			datasets.append(dataset)

		if 12 in features:
			dataset = path + '/Chaos.csv'
			classifical_chaos(preprocessed_fasta, fasta_label[i], 'Yes', dataset)
			datasets.append(dataset)

	"""Concatenating all the extracted features"""

	if datasets:
		dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
		dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
		dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]

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
	ftrain = str(args.fasta_train)
	ftrain_labels = str(args.fasta_label_train)
	ftest = str(args.fasta_test)
	ftest_labels = str(fasta_label_test)
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

	start_time = time.time()

	if os.path.exists(foutput):
		os.remove(foutput)

	# nameseq_test, ftrain, ftest, \
 	# ftrain_labels,ftest_labels = feature_extraction(fasta, fasta_label, features)

	cost = (time.time() - start_time)/60
	print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
