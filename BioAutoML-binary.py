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
# import shutil
import xgboost as xgb
import matplotlib.pyplot as plt #add by Bruno
import shap #add by Bruno
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
#  from sklearn.metrics import multilabel_confusion_matrix
#  from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
#  from sklearn.pipeline import Pipeline
#  from sklearn.preprocessing import MinMaxScaler
#  from sklearn.model_selection import train_test_split
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
from imblearn.under_sampling import NearMiss 
from imblearn.under_sampling import EditedNearestNeighbours 
from imblearn.under_sampling import CondensedNearestNeighbour 
from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek 
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
from yellowbrick.datasets import load_occupancy #add by Bruno
from yellowbrick.features import RadViz #add by Bruno

def header(output_header):

	"""Header Function: Header of the evaluate_model_cross Function"""

	file = open(output_header, 'a')
	file.write('ACC,std_ACC,MCC,std_MCC,F1,std_F1,balanced_ACC,std_balanced_ACC,kappa,std_kappa,gmean,std_gmean')
	file.write('\n')
	return


def save_measures(output_measures, scores):

	"""Save Measures Function: Output of the evaluate_model_cross Function"""

	header(output_measures)
	file = open(output_measures, 'a')
	file.write('%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f' % (scores['test_ACC'].mean(),
				+ scores['test_ACC'].std(), scores['test_MCC'].mean(), scores['test_MCC'].std(),
				+ scores['test_f1'].mean(), scores['test_f1'].std(),
				+ scores['test_ACC_B'].mean(), scores['test_ACC_B'].std(),
				+ scores['test_kappa'].mean(), scores['test_kappa'].std(),
				+ scores['test_gmean'].mean(), scores['test_gmean'].std()))
	file.write('\n')
	return


def evaluate_model_cross(X, y, model, output_cross, matrix_output):

	"""Evaluation Function: Using Cross-Validation"""
	scoring = {'ACC': 'accuracy', 'MCC': make_scorer(matthews_corrcoef), 'f1': 'f1',
			   'ACC_B': 'balanced_accuracy', 'kappa': make_scorer(cohen_kappa_score), 'gmean': make_scorer(geometric_mean_score)}
	kfold = StratifiedKFold(n_splits=10, shuffle=True)
	scores = cross_validate(model, X, LabelEncoder().fit_transform(y), cv=kfold, scoring=scoring)
	save_measures(output_cross, scores)
	y_pred = cross_val_predict(model, X, y, cv=kfold)
	conf_mat = (pd.crosstab(y, y_pred, rownames=['REAL'], colnames=['PREDITO'], margins=True))
	conf_mat.to_csv(matrix_output)
	return


def tuning_rf_ga():

	"""Tuning of classifier using Genetic Algorithm: Random Forest"""

	n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=50)]
	max_features = ['auto', 'sqrt', 'log2', None]
	criterion = ['gini', 'entropy']
	max_depth = [int(x) for x in np.linspace(10, 300, num=50)]
	min_samples_split = [int(x) for x in np.linspace(2, 10, num=8)]
	min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=9)]
	bootstrap = [True, False]

	rf_parameters = {'n_estimators': n_estimators, 'criterion': criterion, 'max_depth': max_depth,
					 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
					 'max_features': max_features, 'bootstrap': bootstrap}

	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	tpot_tuning = TPOTClassifier(generations=20, population_size=10, offspring_size=4,
									 early_stop=12, config_dict={'sklearn.ensemble.RandomForestClassifier': rf_parameters},
									 cv=kfold, scoring=make_scorer(balanced_accuracy_score), n_jobs=n_cpu)
	tpot_tuning.fit(train, train_labels)
	return tpot_tuning


def objective_rf(space):

	"""Tuning of classifier: Objective Function - Random Forest - Bayesian Optimization"""

	model = RandomForestClassifier(n_estimators=int(space['n_estimators']),
								   criterion=space['criterion'],
								   max_depth=int(space['max_depth']),
								   max_features=space['max_features'],
								   min_samples_leaf=int(space['min_samples_leaf']),
								   min_samples_split=int(space['min_samples_split']),
								   random_state=63,
								   bootstrap=space['bootstrap'],
								   n_jobs=n_cpu)

	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	balanced_accuracy = cross_val_score(model,
										train,
										train_labels,
										cv=kfold,
										scoring=make_scorer(balanced_accuracy_score),
										n_jobs=n_cpu).mean()

	return {'loss': -balanced_accuracy, 'status': STATUS_OK}


def tuning_rf_bayesian():

	"""Tuning of classifier: Random Forest - Bayesian Optimization"""

	param = {'criterion': ['entropy', 'gini'], 'max_features': ['auto', 'sqrt', 'log2', None], 'bootstrap': [True, False]}
	space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
			 'n_estimators': hp.quniform('n_estimators', 100, 2000, 50),
			 'max_depth': hp.quniform('max_depth', 10, 100, 5),
			 'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
			 'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
			 'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
			 'bootstrap': hp.choice('bootstrap', [True, False])}

	trials = Trials()
	best_tuning = fmin(fn=objective_rf,
				space=space,
				algo=tpe.suggest,
				max_evals=100,
				trials=trials)

	best_rf = RandomForestClassifier(n_estimators=int(best_tuning['n_estimators']),
									 criterion=param['criterion'][best_tuning['criterion']],
									 max_depth=int(best_tuning['max_depth']),
									 max_features=param['max_features'][best_tuning['max_features']],
									 min_samples_leaf=int(best_tuning['min_samples_leaf']),
									 min_samples_split=int(best_tuning['min_samples_split']),
									 random_state=63,
									 bootstrap=param['bootstrap'][best_tuning['bootstrap']],
									 n_jobs=n_cpu)
	return best_tuning, best_rf


def objective_cb(space):

	"""Tuning of classifier: Objective Function - CatBoost - Bayesian Optimization"""

	model = CatBoostClassifier(n_estimators=int(space['n_estimators']),
							   max_depth=int(space['max_depth']),
							   learning_rate=space['learning_rate'],
							   thread_count=n_cpu, nan_mode='Max', logging_level='Silent',
							   random_state=63)

	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	balanced_accuracy = cross_val_score(model,
										train,
										train_labels,
										cv=kfold,
										scoring=make_scorer(balanced_accuracy_score),
										n_jobs=n_cpu).mean()

	return {'loss': -balanced_accuracy, 'status': STATUS_OK}


def tuning_catboost_bayesian():

	"""Tuning of classifier: CatBoost - Bayesian Optimization"""

	space = {'n_estimators': hp.quniform('n_estimators', 100, 2000, 50),
			 'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
			 'max_depth': hp.quniform('max_depth', 1, 16, 1),
			 #  'random_strength': hp.loguniform('random_strength', 1e-9, 10),
			 #  'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 1.0),
			 #  'border_count': hp.quniform('border_count', 1, 255, 1),
			 #  'l2_leaf_reg': hp.quniform('l2_leaf_reg', 2, 30, 1),
			 #  'scale_pos_weight': hp.uniform('scale_pos_weight', 0.01, 1.0),
			 #  'bootstrap_type' =  hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
	}

	trials = Trials()
	best_tuning = fmin(fn=objective_cb,
					   space=space,
					   algo=tpe.suggest,
					   max_evals=100,
					   trials=trials)

	best_cb = CatBoostClassifier(n_estimators=int(best_tuning['n_estimators']),
								 max_depth=int(best_tuning['max_depth']),
								 learning_rate=best_tuning['learning_rate'],
								 thread_count=n_cpu, nan_mode='Max', logging_level='Silent',
								 random_state=63)

	return best_tuning, best_cb


def objective_lightgbm(space):

	"""Tuning of classifier: Objective Function - Lightgbm - Bayesian Optimization"""

	model = lgb.LGBMClassifier(n_estimators=int(space['n_estimators']),
							   max_depth=int(space['max_depth']),
							   num_leaves=int(space['num_leaves']),
							   learning_rate=space['learning_rate'],
							   subsample=space['subsample'],
							   n_jobs=n_cpu,
							   random_state=63)

	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	balanced_accuracy = cross_val_score(model,
										train,
										train_labels,
										cv=kfold,
										scoring=make_scorer(balanced_accuracy_score),
										n_jobs=n_cpu).mean()

	return {'loss': -balanced_accuracy, 'status': STATUS_OK}


def tuning_lightgbm_bayesian():

	"""Tuning of classifier: Lightgbm - Bayesian Optimization"""

	space = {'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
			 'max_depth': hp.quniform('max_depth', 1, 30, 1),
			 'num_leaves': hp.quniform('num_leaves', 10, 200, 1),
			 'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
			 'subsample': hp.uniform('subsample', 0.1, 1.0)}

	trials = Trials()
	best_tuning = fmin(fn=objective_lightgbm,
					   space=space,
					   algo=tpe.suggest,
					   max_evals=100,
					   trials=trials)

	best_cb = lgb.LGBMClassifier(n_estimators=int(best_tuning['n_estimators']),
								 max_depth=int(best_tuning['max_depth']),
								 num_leaves=int(best_tuning['num_leaves']),
								 learning_rate=best_tuning['learning_rate'],
								 subsample=best_tuning['subsample'],
								 n_jobs=n_cpu,
								 random_state=63)

	return best_tuning, best_cb


def objective_feature_selection(space):

	"""Feature Importance-based Feature selection: Objective Function - Bayesian Optimization"""

	t = space['threshold']

	fs = SelectFromModel(clf, threshold=t)
	fs.fit(train, train_labels)
	fs_train = fs.transform(train)
	kfold = StratifiedKFold(n_splits=5, shuffle=True)
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
					   max_evals=100,
					   trials=trials)

	return best_threshold['threshold']


def feature_importance_fs(model, train, train_labels, column_train):

	"""threshold: features that have an importance of more than ..."""

	if len(column_train) > 100:
		samples = round(int(len(column_train)) * 0.40)
	else:
		samples = round(int(len(column_train)) * 0.80)
	model.fit(train, train_labels)
	importances = set(model.feature_importances_)
	threshold = random.sample(importances, samples)
	best_t = 0
	best_baac = 0
	for t in threshold:
		if t != max(importances):
			fs = SelectFromModel(model, threshold=t)
			fs.fit(train, train_labels)
			fs_train = fs.transform(train)
			kfold = StratifiedKFold(n_splits=5, shuffle=True)
			bacc = cross_val_score(model,
								   fs_train,
								   train_labels,
								   cv=kfold,
								   scoring=make_scorer(balanced_accuracy_score),
								   n_jobs=n_cpu).mean()
			if bacc > best_baac:
				best_baac = bacc
				best_t = t
			elif bacc == best_baac and t > best_t:
				best_t = t
			else:
				pass
		else:
			pass
	return best_t, best_baac


def features_importance_ensembles(model, features, output_importances):

	"""Generate feature importance values"""

	file = open(output_importances, 'a')
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	names = [features[i] for i in indices]
	for f in range(len(features)):
		file.write('%d. Feature (%s): (%f)' % (f + 1, names[f], importances[indices[f]]))
		file.write('\n')
		#  print('%d. %s: (%f)' % (f + 1, names[f], importances[indices[f]]))
	return names


def imbalanced_techniques(model, tech, train, train_labels):

	"""Testing imbalanced data techniques"""

	sm = tech
	pipe = Pipeline([('tech', sm), ('classifier', model)])
	#  train_new, train_labels_new = sm.fit_sample(train, train_labels)
	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	acc = cross_val_score(pipe,
						  train,
						  train_labels,
						  cv=kfold,
						  scoring=make_scorer(balanced_accuracy_score),
						  n_jobs=n_cpu).mean()
	return acc


def imbalanced_function(clf, train, train_labels):

	"""Preprocessing: Imbalanced datasets"""

	print('Checking for imbalanced labels...')
	df = pd.DataFrame(train_labels)
	n_labels = pd.value_counts(df.values.flatten())
	if all(x == n_labels[0] for x in n_labels) is False:
		print('There are imbalanced labels...')
		print('Checking the best technique...')
		performance = []
		smote = imbalanced_techniques(clf, SMOTE(random_state=42), train, train_labels)
		random = imbalanced_techniques(clf, RandomUnderSampler(random_state=42), train, train_labels)
		hybrid_one = imbalanced_techniques(clf, SMOTEENN(random_state=42), train, train_labels)
		hybrid_two = imbalanced_techniques(clf, SMOTETomek(random_state=42), train, train_labels)
		cluster = imbalanced_techniques(clf, ClusterCentroids(random_state=42), train, train_labels)
		near = imbalanced_techniques(clf, EditedNearestNeighbours(), train, train_labels)
		near_miss = imbalanced_techniques(clf, NearMiss(), train, train_labels)
		performance.append(smote)
		performance.append(random)
		performance.append(hybrid_one)
		performance.append(hybrid_two)
		performance.append(cluster)
		performance.append(near)
		performance.append(near_miss)
		max_pos = performance.index(max(performance))
		# print(performance)
		# print(max_pos)
		if max_pos == 0:
			print('Applying Smote - Oversampling...')
			sm = SMOTE(random_state=42)
			train, train_labels = sm.fit_sample(train, train_labels)
		elif max_pos == 1:
			print('Applying Random - Undersampling...')
			sm = RandomUnderSampler(random_state=42)
			train, train_labels = sm.fit_sample(train, train_labels)
		elif max_pos == 2:
			print('Applying SMOTEENN - Hybrid...')
			sm = SMOTEENN(random_state=42)
			train, train_labels = sm.fit_sample(train, train_labels)
		elif max_pos == 3:
			print('Applying SMOTETomek - Hybrid...')
			sm = SMOTETomek(random_state=42)
			train, train_labels = sm.fit_sample(train, train_labels)
		elif max_pos == 4:
			print('Applying ClusterCentroids - Undersampling...')
			sm = ClusterCentroids(random_state=42)
			train, train_labels = sm.fit_sample(train, train_labels)
		elif max_pos == 5:
			print('Applying EditedNearestNeighbours - Undersampling...')
			sm = EditedNearestNeighbours()
			train, train_labels = sm.fit_sample(train, train_labels)
		else:
			print('Applying NearMiss - Undersampling...')
			sm = NearMiss()
			train, train_labels = sm.fit_sample(train, train_labels)
	else:
		print('There are no imbalanced labels...')
	return train, train_labels


def save_prediction(prediction, nameseqs, pred_output):

	"""Saving prediction - test set"""

	file = open(pred_output, 'a')

	if os.path.exists(nameseq_test) is True:
		for i in range(len(prediction)):
			file.write('%s,' % str(nameseqs[i]))
			file.write('%s' % str(prediction[i]))
			file.write('\n')
	else:
		for i in range(len(prediction)):
			file.write('%s' % str(prediction[i]))
			file.write('\n')
	return
#add by Bruno
def type_model(explainer, model, data):
	shap_values = explainer(data)
	catype = "<class 'lightgbm.sklearn.LGBMClassifier'>"
	randtype = "<class 'sklearn.ensemble._forest.RandomForestClassifier'>"
	if catype == str(type(model) ) or randtype == str(type(model)):
		shap_values = shap_values[:, :, 0]
	return shap_values         
def inter_shap_waterf(explainer, X_train, X_label,model,output):
	X_label= pd.DataFrame(data={'label': X_label}) 
	classes = X_label.iloc[:,0].unique()
	graph_name = 'waterfall'
	for i in range(len(X_label.iloc[:,0].unique())):
		for j in range(2):
			subset = X_train[X_label.label==classes[i]]
			shap_valuesW = type_model(explainer, model, subset)
			sp = shap.plots.waterfall(shap_valuesW[random.randrange(subset.shape[0])], show=False)
			waterfall_name = output + graph_name + classes[i] + str(j) + '.png'
			plt.savefig(waterfall_name, dpi=300,bbox_inches='tight')
			plt.close(sp)
def interp_shap(model, X_train, X_label,output):
    #to create a visualizer
	explainer = shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
	shap.initjs()
	shap_values = type_model(explainer, model, X_train)
    #bar graph
	sp = shap.plots.bar(shap_values, show=False)
	namefig = output + 'bar_graph.png'
	plt.savefig(namefig, dpi=300,bbox_inches='tight')
	plt.close(sp)
    #beeswarm graph
	sp = shap.plots.beeswarm(shap_values, show=False)
	namefig = output + 'beeswarm_graph.png'
	plt.savefig(namefig, dpi=300,bbox_inches='tight')
	plt.close(sp)
    #scatter graph
	feats = random.sample(range(0,X_train.shape[1]), 2)
	sp = shap.plots.scatter(shap_values[:,feats[0]], color=shap_values, show=False)
	namefig = output + 'scatter_graph1.png'
	plt.savefig(namefig)
	plt.close(sp)
	sp = shap.plots.scatter(shap_values[:,feats[1]], color=shap_values, show=False)
	namefig = output + 'scatter_graph2.png'
	plt.savefig(namefig)
	plt.close(sp)
    #waterfall graph
	inter_shap_waterf(explainer, X_train, X_label,model,output)
def interp_yellow(model,X_train,X_label,output):
	X_label= pd.DataFrame(data={'label': X_label})
	classes = X_label.iloc[:,0].unique()
	visualizer = RadViz(classes=classes)
	labelencoder = LabelEncoder()
	X_label = labelencoder.fit_transform(X_label)
	feats = random.sample(range(0,X_train.shape[1]), 5)
	visualizer.fit(X_train.iloc[:,[i for i in feats]], X_label)
	visualizer.transform(X_train.iloc[:,[i for i in feats]])
	namefig = output + 'radial_graph.png'
	plt.savefig(namefig)
#add by Bruno

def binary_pipeline(test, test_labels, test_nameseq, norm, fs, classifier, tuning, output):

	global clf, train, train_labels

	if not os.path.exists(output):
		os.mkdir(output)

	train = train_read
	train_labels = train_labels_read
	column_train = train.columns
	column_test = ''
	output = output + '/'

	#  tmp = sys.stdout
	#  log_file = open(output + 'task.log', 'a')
	#  sys.stdout = log_file

	"""Number of Samples and Features: Train and Test"""

	print('Number of samples (train): ' + str(len(train)))

	if os.path.exists(ftest) is True:
		column_test = test.columns
		print('Number of samples (test): ' + str(len(test)))

	print('Number of features (train): ' + str(len(column_train)))

	if os.path.exists(ftest_labels) is True:
		print('Number of features (test): ' + str(len(column_test)))

	"""Preprocessing:  Missing Values"""

	print('Checking missing values...')
	missing = train.isnull().values.any()
	inf = train.isin([np.inf, -np.inf]).values.any()
	missing_test = False
	inf_test = False
	if os.path.exists(ftest) is True:
		missing_test = test.isnull().values.any()
		inf_test = test.isin([np.inf, -np.inf]).values.any()
	if missing or inf or missing_test or inf_test:
		print('There are missing values...')
		print('Applying SimpleImputer - strategy (mean)...')
		train.replace([np.inf, -np.inf], np.nan, inplace=True)
		imp = SimpleImputer(missing_values=np.nan, strategy='mean')
		train = pd.DataFrame(imp.fit_transform(train), columns=column_train)
		if os.path.exists(ftest) is True:
			test.replace([np.inf, -np.inf], np.nan, inplace=True)
			test = pd.DataFrame(imp.transform(test), columns=column_test)
		else:
			pass
	else:
		print('There are no missing values...')

	"""Preprocessing: StandardScaler()"""

	if norm is True:
		print('Applying StandardScaler()....')
		sc = StandardScaler()
		train = pd.DataFrame(sc.fit_transform(train), columns=column_train)
		if os.path.exists(ftest) is True:
			test = pd.DataFrame(sc.transform(test), columns=column_test)
		else:
			pass

	"""Choosing Classifier """

	print('Choosing Classifier...')
	if classifier == 0:
		if tuning is True:
			print('Tuning: ' + str(tuning))
			print('Classifier: CatBoost')
			clf = CatBoostClassifier(n_estimators=500, thread_count=n_cpu, nan_mode='Max',
									 logging_level='Silent', random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
			best_tuning, clf = tuning_catboost_bayesian()
			print('Finished Tuning')
		else:
			print('Tuning: ' + str(tuning))
			print('Classifier: CatBoost')
			clf = CatBoostClassifier(n_estimators=500, thread_count=n_cpu, nan_mode='Max',
									 logging_level='Silent', random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
	elif classifier == 1:
		if tuning is True:
			print('Tuning: ' + str(tuning))
			print('Classifier: Random Forest')
			clf = RandomForestClassifier(n_estimators=200, n_jobs=n_cpu, random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
			best_tuning, clf = tuning_rf_bayesian()
			print('Finished Tuning')
		else:
			print('Tuning: ' + str(tuning))
			print('Classifier: Random Forest')
			clf = RandomForestClassifier(n_estimators=200, n_jobs=n_cpu, random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
	elif classifier == 2:
		if tuning is True:
			print('Tuning: ' + str(tuning))
			print('Classifier: LightGBM')
			clf = lgb.LGBMClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
			best_tuning, clf = tuning_lightgbm_bayesian()
			print('Finished Tuning')
		else:
			print('Tuning: ' + str(tuning))
			print('Classifier: LightGBM')
			clf = lgb.LGBMClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
	elif classifier == 3:
		if tuning is True:
			print('Tuning: ' + str(tuning))
			print('Classifier: XGBClassifier')
			clf = xgb.XGBClassifier(eval_metric='mlogloss', random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
			print('Tuning not yet available for XGBClassifier.')
		else:
			print('Tuning: ' + str(tuning))
			print('Classifier: XGBClassifier')
			clf = xgb.XGBClassifier(eval_metric='mlogloss', random_state=63)
			if imbalance_data is True:
				train, train_labels = imbalanced_function(clf, train, train_labels)
	else:
		sys.exit('This classifier option does not exist - Try again')

	"""Preprocessing: Feature Importance-Based Feature Selection"""

	feature_name = column_train
	if fs == 1:
		print('Applying Feature Importance-Based Feature Selection...')
		# best_t, best_baac = feature_importance_fs(clf, train, train_labels, column_train)
		best_t = feature_importance_fs_bayesian(clf, train, train_labels)
		fs = SelectFromModel(clf, threshold=best_t)
		fs.fit(train, train_labels)
		feature_idx = fs.get_support()
		feature_name = column_train[feature_idx]
		train = pd.DataFrame(fs.transform(train), columns=feature_name)
		if os.path.exists(ftest) is True:
			test = pd.DataFrame(fs.transform(test), columns=feature_name)
		else:
			pass
		print('Best Feature Subset: ' + str(len(feature_name)))
		print('Reduction: ' + str(len(column_train)-len(feature_name)) + ' features')
		fs_train = output + 'best_feature_train.csv'
		fs_test = output + 'best_feature_test.csv'
		print('Saving dataset with selected feature subset - train: ' + fs_train)
		train.to_csv(fs_train, index=False)
		if os.path.exists(ftest) is True:
			print('Saving dataset with selected feature subset - test: ' + fs_test)
			test.to_csv(fs_test, index=False)
		print('Feature Selection - Finished...')

	"""Training - StratifiedKFold (cross-validation = 10)..."""

	print('Training: StratifiedKFold (cross-validation = 10)...')
	train_output = output + 'training_kfold(10)_metrics.csv'
	matrix_output = output + 'training_confusion_matrix.csv'
	model_output = output + 'trained_model.sav'
	evaluate_model_cross(train, train_labels, clf, train_output, matrix_output)
	clf.fit(train, train_labels)
	joblib.dump(clf, model_output)
	print('Saving results in ' + train_output + '...')
	print('Saving confusion matrix in ' + matrix_output + '...')
	print('Saving trained model in ' + model_output + '...')
	print('Training: Finished...')

	interp_shap(clf, train, train_labels,output) #add by Bruno
	interp_yellow(clf, train, train_labels,output) #add by Bruno
	
	"""Generating Feature Importance - Selected feature subset..."""

	print('Generating Feature Importance - Selected feature subset...')
	importance_output = output + 'feature_importance.csv'
	features_importance_ensembles(clf, feature_name, importance_output)
	print('Saving results in ' + importance_output + '...')

	"""Testing model..."""

	if os.path.exists(ftest) is True:
		print('Generating Performance Test...')
		preds = clf.predict(test)
		pred_output = output + 'test_predictions.csv'
		print('Saving prediction in ' + pred_output + '...')
		save_prediction(preds, test_nameseq, pred_output)
		if os.path.exists(ftest_labels) is True:
			print('Generating Metrics - Test set...')
			labels = np.unique(test_labels)
			accu = accuracy_score(test_labels, preds)
			recall = recall_score(test_labels, preds, pos_label=labels[0])
			precision = precision_score(test_labels, preds, pos_label=labels[0])
			f1 = f1_score(test_labels, preds, pos_label=labels[0])
			auc = roc_auc_score(test_labels, clf.predict_proba(test)[:, 1])
			balanced = balanced_accuracy_score(test_labels, preds)
			gmean = geometric_mean_score(test_labels, preds)
			mcc = matthews_corrcoef(test_labels, preds)
			matrix_test = (pd.crosstab(test_labels, preds, rownames=["REAL"], colnames=["PREDITO"], margins=True))

			metrics_output = output + 'metrics_test.csv'
			print('Saving Metrics - Test set: ' + metrics_output + '...')
			file = open(metrics_output, 'a')
			file.write('Metrics: Test Set')
			file.write('\n')
			file.write('Accuracy: %s' % accu)
			file.write('\n')
			file.write('Recall: %s' % recall)
			file.write('\n')
			file.write('Precision: %s' % precision)
			file.write('\n')
			file.write('F1: %s' % f1)
			file.write('\n')
			file.write('AUC: %s' % auc)
			file.write('\n')
			file.write('balanced ACC: %s' % balanced)
			file.write('\n')
			file.write('gmean: %s' % gmean)
			file.write('\n')
			file.write('MCC: %s' % mcc)
			file.write('\n')

			matrix_output_test = output + 'test_confusion_matrix.csv'
			matrix_test.to_csv(matrix_output_test)
			print('Saving confusion matrix in ' + matrix_output_test + '...')
			print('Task completed - results generated in ' + output + '!')

		else:
			print('There are no test labels for evaluation, check parameters...')
			#  sys.stdout = tmp
			#  log_file.close()
	else:
		print('There are no test sequences for evaluation, check parameters...')
		print('Task completed - results generated in ' + output + '!')
		#  sys.stdout = tmp
		#  log_file.close()

	return


##########################################################################
##########################################################################
if __name__ == '__main__':
	print('\n')
	print('###################################################################################')
	print('###################################################################################')
	print('#####################        BioAutoML - Binary             #######################')
	print('##########              Author: Robson Parmezan Bonidia                 ###########')
	print('##########         WebPage: https://bonidia.github.io/website/          ###########')
	print('###################################################################################')
	print('###################################################################################')
	print('\n')
	parser = argparse.ArgumentParser()
	parser.add_argument('-train', '--train', help='csv format file, e.g., train.csv')
	parser.add_argument('-train_label', '--train_label', default='', help='csv format file, e.g., labels.csv')
	parser.add_argument('-test', '--test', help='csv format file, e.g., train.csv')
	parser.add_argument('-test_label', '--test_label', default='', help='csv format file, e.g., labels.csv')
	parser.add_argument('-test_nameseq', '--test_nameseq', default='', help='csv with sequence names')
	parser.add_argument('-nf', '--normalization', type=bool, default=False,
						help='Normalization - Features (default = False)')
	parser.add_argument('-fs', '--featureselection', default=1,
						help='Feature Selection (default = True)')
	parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
	parser.add_argument('-classifier', '--classifier', default=0,
						help='Classifier - 0: CatBoost, 1: Random Forest'
							 '2: LightGBM, 3: XGBoost')
	parser.add_argument('-imbalance', '--imbalance', type=bool, default=False,
						help='To deal with the imbalanced dataset problem - True = Yes, False = No, '
							 'default = False')
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
	fs = int(args.featureselection)
	n_cpu = int(args.n_cpu)
	classifier = int(args.classifier)
	imbalance_data = args.imbalance
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

	binary_pipeline(test_read, test_labels_read, test_nameseq_read, norm, fs, classifier, tuning, foutput)
	cost = (time.time() - start_time)/60
	print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
