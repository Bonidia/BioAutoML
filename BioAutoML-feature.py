import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import argparse
import sys
import os.path
import time
import lightgbm as lgb
import joblib
from sklearn.model_selection import cross_val_predict
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


def save_files():

	"""In constrution"""


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

	path = 'feat_extraction'
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
	parser.add_argument('-fasta_train','--fasta_train',nargs='+',
						help='fasta format file, e.g., example_fasta/Ecoli_K12.fasta')
	parser.add_argument('-fasta_label_train','--fasta_label_train',nargs='+',
						help='labels for fasta files, e.g., sRNA circRNA lncRNA')
	parser.add_argument('-fasta_test', '--fasta', nargs='+',
						help='fasta format file, e.g., example_fasta/Ecoli_K12.fasta')
	parser.add_argument('-fasta_label_test', '--fasta_test', nargs='+',
						help='labels for fasta files, e.g., sRNA circRNA lncRNA')
	parser.add_argument('-nf', '--normalization', type=bool, default=False,
						help='Normalization - Features (default = False)')
	parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
	parser.add_argument('-classifier', '--classifier', default=0,
						help='Classifier - 0: CatBoost, 1: Random Forest'
							 '2: LightGBM, 3: All classifiers - choose the best')
	parser.add_argument('-sampling', '--sampling', default='False',
						help='Apply oversampling or undersampling techniques - True = Yes,'
							 'False = No, default = False')
	parser.add_argument('-tuning', '--tuning_classifier', type=bool, default=False,
						help='Tuning Classifier - True = Yes, False = No, default = False')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	ftrain = str(args.train)
	ftrain_labels = str(args.train_label)
	ftest = str(args.test)
	ftest_labels = str(args.test_label)
	nameseq_test = str(args.test_nameseq)
	norm = args.normalization
	n_cpu = int(args.n_cpu)
	classifier = int(args.classifier)
	tuning = args.tuning_classifier
	foutput = str(args.output)
	start_time = time.time()

	if os.path.exists(ftrain) is True:
		train_read = pd.read_csv(ftrain)
		print('Train - %s: Found File' % ftrain)
	else:
		print('Train - %s: File not exists' % ftrain)
		sys.exit()

	if os.path.exists(ftrain_labels) is True:
		train_labels_read = pd.read_csv(ftrain_labels).values.ravel()
		print('Train_labels - %s: Found File' % ftrain_labels)
	else:
		print('Train_labels - %s: File not exists' % ftrain_labels)
		sys.exit()

	test_read = ''
	if ftest != '':
		if os.path.exists(ftest) is True:
			test_read = pd.read_csv(ftest)
			print('Test - %s: Found File' % ftest)
		else:
			print('Test - %s: File not exists' % ftest)
			sys.exit()

	test_labels_read = ''
	if ftest_labels != '':
		if os.path.exists(ftest_labels) is True:
			test_labels_read = pd.read_csv(ftest_labels).values.ravel()
			print('Test_labels - %s: Found File' % ftest_labels)
		else:
			print('Test_labels - %s: File not exists' % ftest_labels)
			sys.exit()

	test_nameseq_read = ''
	if nameseq_test != '':
		if os.path.exists(nameseq_test) is True:
			test_nameseq_read = pd.read_csv(nameseq_test).values.ravel()
			print('Test_nameseq - %s: Found File' % nameseq_test)
		else:
			print('Test_nameseq - %s: File not exists' % nameseq_test)
			sys.exit()

	binary_pipeline(test_read, test_labels_read, test_nameseq_read, norm, classifier, tuning, foutput)
	cost = (time.time() - start_time)/60
	print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
