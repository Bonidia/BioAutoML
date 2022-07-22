import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import argparse
import subprocess
import shutil
import sys
import os.path
import time
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Testing
# python BioAutoML-feature.py
# -fasta_train Case\ Studies/CS-II/train/miRNA.fasta
# Case\ Studies/CS-II/train/pre_miRNA.fasta
# Case\ Studies/CS-II/train/tRNA.fasta
# -fasta_label_train miRNA pre_miRNA tRNA
# -fasta_test Case\ Studies/CS-II/test/miRNA.fasta
# Case\ Studies/CS-II/test/pre_miRNA.fasta
# Case\ Studies/CS-II/test/tRNA.fasta
# -fasta_label_test miRNA pre_miRNA tRNA
# -output results/


def objective_rf(space):

	"""Automated Feature Engineering - Objective Function - Bayesian Optimization"""

	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464))}

	for descriptor, ind in descriptors.items():
		if int(space[descriptor]) == 1:
			index = index + ind

	x = df_x.iloc[:, index]

	# print(index)

	if int(space['Classifier']) == 0:
		if len(fasta_label_train) > 2:
			model = AdaBoostClassifier(random_state=63)
		else:
			model = CatBoostClassifier(n_estimators=500,
									   thread_count=n_cpu, nan_mode='Max',
								   	   logging_level='Silent', random_state=63)
	elif int(space['Classifier']) == 1:
		model = RandomForestClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)
	else:
		model = lgb.LGBMClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)

	# print(model)

	if len(fasta_label_train) > 2:
		score = make_scorer(f1_score, average='weighted')
	else:
		score = make_scorer(balanced_accuracy_score)

	kfold = StratifiedKFold(n_splits=10, shuffle=True)
	metric = cross_val_score(model,
							 x,
							 labels_y,
							 cv=kfold,
							 scoring=score,
							 n_jobs=n_cpu).mean()

	return {'loss': -metric, 'status': STATUS_OK}


def feature_engineering(estimations, train, train_labels, test, foutput):

	"""Automated Feature Engineering - Bayesian Optimization"""

	global df_x, labels_y

	print('Automated Feature Engineering - Bayesian Optimization')

	df_x = pd.read_csv(train)
	labels_y = pd.read_csv(train_labels)

	if test != '':
		df_test = pd.read_csv(test)

	path_bio = foutput + '/best_descriptors'
	if not os.path.exists(path_bio):
		os.mkdir(path_bio)

	param = {'NAC': [0, 1], 'DNC': [0, 1],
			 'TNC': [0, 1], 'kGap_di': [0, 1], 'kGap_tri': [0, 1],
			 'ORF': [0, 1], 'Fickett': [0, 1],
			 'Shannon': [0, 1], 'FourierBinary': [0, 1],
			 'FourierComplex': [0, 1], 'Tsallis': [0, 1],
			 'Classifier': [0, 1, 2]}

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
			 'Classifier': hp.choice('Classifier', [0, 1, 2])}

	trials = Trials()
	best_tuning = fmin(fn=objective_rf,
				space=space,
				algo=tpe.suggest,
				max_evals=estimations,
				trials=trials)

	index = list()
	descriptors = {'NAC': list(range(0, 4)), 'DNC': list(range(4, 20)),
				   'TNC': list(range(20, 84)), 'kGap_di': list(range(84, 148)),
				   'kGap_tri': list(range(148, 404)), 'ORF': list(range(404, 414)),
				   'Fickett': list(range(414, 416)), 'Shannon': list(range(416, 421)),
				   'FourierBinary': list(range(421, 440)), 'FourierComplex': list(range(440, 459)),
				   'Tsallis': list(range(459, 464))}

	for descriptor, ind in descriptors.items():
		result = param[descriptor][best_tuning[descriptor]]
		if result == 1:
			index = index + ind

	classifier = param['Classifier'][best_tuning['Classifier']]

	btrain = df_x.iloc[:, index]
	path_btrain = path_bio + '/best_train.csv'
	btrain.to_csv(path_btrain, index=False, header=True)

	if test != '':
		btest = df_test.iloc[:, index]
		path_btest = path_bio + '/best_test.csv'
		btest.to_csv(path_btest, index=False, header=True)
	else:
		btest, path_btest = '', ''

	return classifier, path_btrain, path_btest, btrain, btest


def feature_extraction(ftrain, ftrain_labels, ftest, ftest_labels, features, foutput):

	"""Extracts the features from the sequences in the fasta files."""

	path = foutput + '/feat_extraction'
	path_results = foutput

	try:
		shutil.rmtree(path)
		shutil.rmtree(path_results)
	except OSError as e:
		print("Error: %s - %s." % (e.filename, e.strerror))
		print('Creating Directory...')

	if not os.path.exists(path_results):
		os.mkdir(path_results)

	if not os.path.exists(path):
		os.mkdir(path)
		os.mkdir(path + '/train')
		os.mkdir(path + '/test')

	labels = [ftrain_labels]
	fasta = [ftrain]
	train_size = 0

	if fasta_test:
		labels.append(ftest_labels)
		fasta.append(ftest)

	datasets = []
	fasta_list = []

	print('Extracting features with MathFeature...')

	for i in range(len(labels)):
		for j in range(len(labels[i])):
			file = fasta[i][j].split('/')[-1]
			if i == 0:  # Train
				preprocessed_fasta = path + '/train/pre_' + file
				subprocess.run(['python', 'MathFeature/preprocessing/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				train_size += len([1 for line in open(preprocessed_fasta) if line.startswith(">")])
			else:  # Test
				preprocessed_fasta = path + '/test/pre_' + file
				subprocess.run(['python', 'MathFeature/preprocessing/preprocessing.py',
								'-i', fasta[i][j], '-o', preprocessed_fasta],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

			fasta_list.append(preprocessed_fasta)

			if 1 in features:
				dataset = path + '/NAC.csv'
				subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py',
								'-i', preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'NAC', '-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 2 in features:
				dataset = path + '/DNC.csv'
				subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'DNC', '-seq', '1'], stdout=subprocess.DEVNULL,
								stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 3 in features:
				dataset = path + '/TNC.csv'
				subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-t', 'TNC', '-seq', '1'], stdout=subprocess.DEVNULL,
								stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 4 in features:
				dataset_di = path + '/kGap_di.csv'
				dataset_tri = path + '/kGap_tri.csv'

				subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', dataset_di, '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '2', '-seq', '1'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

				subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
								preprocessed_fasta, '-o', dataset_tri, '-l',
								labels[i][j], '-k', '1', '-bef', '1',
								'-aft', '3', '-seq', '1'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset_di)
				datasets.append(dataset_tri)

			if 5 in features:
				dataset = path + '/ORF.csv'
				subprocess.run(['python', 'MathFeature/methods/CodingClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j]],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 6 in features:
				dataset = path + '/Fickett.csv'
				subprocess.run(['python', 'MathFeature/methods/FickettScore.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-seq', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 7 in features:
				dataset = path + '/Shannon.csv'
				subprocess.run(['python', 'MathFeature/methods/EntropyClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-k', '5', '-e', 'Shannon'],
								stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 8 in features:
				dataset = path + '/FourierBinary.csv'
				subprocess.run(['python', 'MathFeature/methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-r', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 9 in features:
				dataset = path + '/FourierComplex.csv'
				subprocess.run(['python', 'other-methods/FourierClass.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-r', '6'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

			if 10 in features:
				dataset = path + '/Tsallis.csv'
				subprocess.run(['python', 'other-methods/TsallisEntropy.py', '-i',
								preprocessed_fasta, '-o', dataset, '-l', labels[i][j],
								'-k', '5', '-q', '2.3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				datasets.append(dataset)

	if 11 in features:
		dataset = path + '/Chaos.csv'
		# classifical_chaos(preprocessed_fasta, labels[i][j], 'Yes', dataset)
		datasets.append(dataset)

	if 12 in features:
		dataset = path + '/BinaryMapping.csv'

		labels_list = ftrain_labels + ftest_labels
		text_input = ''
		for i in range(len(fasta_list)):
			text_input += fasta_list[i] + '\n' + labels_list[i] + '\n'

		subprocess.run(['python', 'MathFeature/methods/MappingClass.py',
						'-n', str(len(fasta_list)), '-o',
						dataset, '-r', '1'], text=True, input=text_input,
					   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

		with open(dataset, 'r') as temp_f:
			col_count = [len(l.split(",")) for l in temp_f.readlines()]

		colnames = ['BinaryMapping_' + str(i) for i in range(0, max(col_count))]

		df = pd.read_csv(dataset, names=colnames, header=None)
		df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
		df.to_csv(dataset, index=False)
		
		datasets.append(dataset)

	"""Concatenating all the extracted features"""

	if datasets:
		datasets = list(dict.fromkeys(datasets))
		dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
		dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
		dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]

	X_train = dataframes.iloc[:train_size, :]
	X_train.pop('nameseq')
	y_train = X_train.pop('label')
	ftrain = path + '/ftrain.csv'
	X_train.to_csv(ftrain, index=False)
	flabeltrain = path + '/flabeltrain.csv'
	y_train.to_csv(flabeltrain, index=False, header=True)
	
	fnameseqtest, ftest, flabeltest = '', '', ''

	if fasta_test:
		X_test = dataframes.iloc[train_size:, :]
		y_test = X_test.pop('label')
		nameseq_test = X_test.pop('nameseq')
		fnameseqtest = path + '/fnameseqtest.csv'
		nameseq_test.to_csv(fnameseqtest, index=False, header=True)
		ftest = path + '/ftest.csv'
		X_test.to_csv(ftest, index=False)
		flabeltest = path + '/flabeltest.csv'
		y_test.to_csv(flabeltest, index=False, header=True)

	return fnameseqtest, ftrain, flabeltrain, ftest, flabeltest

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
	parser.add_argument('-estimations', '--estimations', default=50, help='number of estimations - BioAutoML - default = 50')
	parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
	parser.add_argument('-output', '--output', help='results directory, e.g., result/')

	args = parser.parse_args()
	fasta_train = args.fasta_train
	fasta_label_train = args.fasta_label_train
	fasta_test = args.fasta_test
	fasta_label_test = args.fasta_label_test
	estimations = int(args.estimations)
	n_cpu = int(args.n_cpu)
	foutput = str(args.output)

	for fasta in fasta_train:
		if os.path.exists(fasta) is True:
			print('Train - %s: Found File' % fasta)
		else:
			print('Train - %s: File not exists' % fasta)
			sys.exit()

	if fasta_test:
		for fasta in fasta_test:
			if os.path.exists(fasta) is True:
				print('Test - %s: Found File' % fasta)
			else:
				print('Test - %s: File not exists' % fasta)
				sys.exit()

	start_time = time.time()

	features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	fnameseqtest, ftrain, ftrain_labels, \
		ftest, ftest_labels = feature_extraction(fasta_train, fasta_label_train,
												 fasta_test, fasta_label_test, features, foutput)

	classifier, path_train, path_test, train_best, test_best = \
		feature_engineering(estimations, ftrain, ftrain_labels, ftest, foutput)

	cost = (time.time() - start_time) / 60
	print('Computation time - Pipeline - Automated Feature Engineering: %s minutes' % cost)

	if len(fasta_label_train) > 2:
		subprocess.run(['python', 'BioAutoML-multiclass.py', '-train', path_train,
						 '-train_label', ftrain_labels, '-test', path_test,
						 '-test_label', ftest_labels, '-test_nameseq',
						 fnameseqtest, '-nf', 'True', '-classifier', str(classifier),
						 '-n_cpu', str(n_cpu), '-output', foutput])
	else:
		subprocess.run(['python', 'BioAutoML-binary.py', '-train', path_train,
						 '-train_label', ftrain_labels, '-test', path_test, '-test_label',
						 ftest_labels, '-test_nameseq', fnameseqtest,
						 '-nf', 'True', '-classifier', str(classifier), '-n_cpu', str(n_cpu),
						 '-output', foutput])

##########################################################################
##########################################################################
