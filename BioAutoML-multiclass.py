import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import sys
import os.path
import time
import lightgbm as lgb
import joblib
import shap
import xgboost as xgb
# import shutil
from catboost import CatBoostClassifier
from orderedset import OrderedSet
# from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import multilabel_confusion_matrix
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
# from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from interpretability_report import Report, REPORT_MAIN_TITLE_MULTICLASS, REPORT_SHAP_PREAMBLE, \
    REPORT_SHAP_SUMMARY_1, REPORT_SHAP_SUMMARY_2, REPORT_SHAP_WATERFALL_1, REPORT_SUMMARY_TITLE, \
    REPORT_WATERFALL_TITLE, REPORT_SHAP_WATERFALL_2

PLOT_NAME_WATERFALL = 0
PLOT_NAME_SUMMARY = 1

def header(output_header):

    """Header Function: Header of the evaluate_model_cross Function"""

    file = open(output_header, 'a')
    file.write('ACC,std_ACC,MCC,std_MCC,F1_micro,std_F1_micro,'
               'F1_macro,std_F1_macro,F1_w,std_F1_w,kappa,std_kappa')
    file.write('\n')
    return


def save_measures(output_measures, scores):

    """Save Measures Function: Output of the evaluate_model_cross Function"""

    header(output_measures)
    file = open(output_measures, 'a')
    file.write('%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f' % (scores['test_ACC'].mean(),
                + scores['test_ACC'].std(), scores['test_MCC'].mean(), scores['test_MCC'].std(),
                + scores['test_f1_mi'].mean(), scores['test_f1_mi'].std(),
                + scores['test_f1_ma'].mean(), scores['test_f1_ma'].std(),
                + scores['test_f1_w'].mean(), scores['test_f1_w'].std(),
                + scores['test_kappa'].mean(), scores['test_kappa'].std()))
    file.write('\n')
    return


def evaluate_model_cross(X, y, model, output_cross, matrix_output):

    """Evaluation Function: Using Cross-Validation"""

    scoring = {'ACC': make_scorer(accuracy_score),
               'MCC': make_scorer(matthews_corrcoef),
               'f1_mi': make_scorer(f1_score, average='micro'),
               'f1_ma': make_scorer(f1_score, average='macro'),
               'f1_w': make_scorer(f1_score, average='weighted'),
               'kappa': make_scorer(cohen_kappa_score)}

    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_validate(model, X, y, cv=kfold, scoring=scoring)
    save_measures(output_cross, scores)
    y_pred = cross_val_predict(model, X, y, cv=kfold)
    conf_mat = (pd.crosstab(y, y_pred, rownames=['REAL'], colnames=['PREDITO'], margins=True))
    conf_mat.to_csv(matrix_output)


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
    metric = cross_val_score(model,
                             train,
                             train_labels,
                             cv=kfold,
                             scoring=make_scorer(f1_score, average='weighted'),
                             n_jobs=n_cpu).mean()

    return {'loss': -metric, 'status': STATUS_OK}


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


# function not used anywhere
def objective_cb(space):

    """Tuning of classifier: Objective Function - CatBoost - Bayesian Optimization"""

    model = CatBoostClassifier(n_estimators=int(space['n_estimators']),
                               max_depth=int(space['max_depth']),
                               learning_rate=space['learning_rate'],
                               thread_count=n_cpu, nan_mode='Max', 
                               logging_level='Silent', random_state=63)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    metric = cross_val_score(model,
                             train,
                             train_labels,
                             cv=kfold,
                             scoring=make_scorer(f1_score, average='weighted'),
                             n_jobs=n_cpu).mean()

    return {'loss': -metric, 'status': STATUS_OK}


# function not used anywhere
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
                                 thread_count=n_cpu, nan_mode='Max', logging_level='Silent', random_state=63)

    return best_tuning, best_cb


def objective_lightgbm(space):

    """Tuning of classifier: Objective Function - Lightgbm - Bayesian Optimization"""

    model = lgb.LGBMClassifier(n_estimators=int(space['n_estimators']),
                               max_depth=int(space['max_depth']),
                               num_leaves=int(space['num_leaves']),
                               learning_rate=space['learning_rate'],
                               subsample=space['subsample'],
                               n_jobs=n_cpu, random_state=63)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    metric = cross_val_score(model,
                                        train,
                                        train_labels,
                                        cv=kfold,
                                        scoring=make_scorer(f1_score, average='weighted'),
                                        n_jobs=n_cpu).mean()

    return {'loss': -metric, 'status': STATUS_OK}


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
                                 n_jobs=n_cpu, random_state=63)

    return best_tuning, best_cb


def objective_feature_selection(space):

    """Feature Importance-based Feature selection: Objective Function - Bayesian Optimization"""

    t = space['threshold']

    fs = SelectFromModel(clf, threshold=t)
    fs.fit(train, train_labels)
    fs_train = fs.transform(train)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    f1 = cross_val_score(clf,
                           fs_train,
                           train_labels,
                           cv=kfold,
                           scoring=make_scorer(f1_score, average='weighted'),
                           n_jobs=n_cpu).mean()

    return {'loss': -f1, 'status': STATUS_OK}


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
                       max_evals=50,
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
                                   scoring=make_scorer(f1_score, average='weighted'),
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
    f1 = cross_val_score(pipe,
                          train,
                          train_labels,
                          cv=kfold,
                          scoring=make_scorer(f1_score, average='weighted'),
                          n_jobs=n_cpu).mean()
    return f1


def imbalanced_function(clf, train, train_labels):

    """Preprocessing: Imbalanced datasets"""

    print('Checking for imbalanced labels...')
    df = pd.DataFrame(train_labels)
    n_labels = pd.value_counts(df.values.flatten())
    if all(x == n_labels[0] for x in n_labels) is False:
        print('There are imbalanced labels...')
        print('Checking the best technique...')
        smote = imbalanced_techniques(clf, SMOTE(random_state=42), train, train_labels)
        random = imbalanced_techniques(clf, RandomUnderSampler(random_state=42), train, train_labels)
        if smote > random:
            print('Applying Smote - Oversampling...')
            sm = SMOTE(random_state=42)
            train, train_labels = sm.fit_sample(train, train_labels)
        else:
            print('Applying Random - Undersampling...')
            sm = RandomUnderSampler(random_state=42)
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


def randomize_samples(targets, _class, n_samples=1):

    """Get a given number of samples which match with class '_class'"""

    enum_targets = enumerate(targets)
    class_samples = np.array(list(filter(lambda x: x[1] == _class, enum_targets)))

    try:
        chosen = np.random.choice(len(class_samples), size=n_samples, replace=False) 
        return class_samples[chosen,0]
    except ValueError:
        raise ValueError(
            f"Error: There's not enough samples of class {_class} in targets " +\
            f"(n_samples={n_samples} is too high)."
        )
        

def generate_waterfall_plot(id_plot, class_shap_values, class_name, sample_row, base_value,
                            feature_names, path):

    """Generates a waterfall plot for a given sample"""

    local_name = os.path.join(path, f"waterfall_random_{class_name}_{id_plot}.png")

    exp = shap.Explanation(values=class_shap_values, base_values=base_value,
                           data=sample_row, feature_names=feature_names)
    fig = plt.figure()
    plt.title(f'Waterfall plot for sample {id_plot} of class \'{class_name}\'', fontsize=16)
    shap.waterfall_plot(exp, show=False)
    plt.savefig(local_name, bbox_inches = "tight")

    return local_name


def generate_summary_plot(class_shap_values, class_name, data, feature_names, path):
    
    """Generates a summary plot for a given class of the multiclass classification"""
    
    local_name = os.path.join(path, f"summary_{class_name}.png")

    fig = plt.figure()
    plt.title(f'Summary plot for class \'{class_name}\'', fontsize=16)
    shap.summary_plot(class_shap_values, data, feature_names=feature_names, show=False)
    plt.savefig(local_name, bbox_inches = 'tight')

    return local_name


def generate_all_plots(model, features, feature_names, targets, path='explanations', n_samples=3):

    """Used to generate each of the plots used to explain the model's decision"""

    generated_plt = {}

    print("Training the explainer model...")
    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(features))
    print(f"shap_values.shape: {shap_values.shape}")
    print(f"targets.shape: {targets.shape}")
    print(f"features.shape: {features.shape}")
    print("Explainer trained successfully!")

    # SHAP seems to sort the classes before calculating the shapley values matrix for each class,
    #   then it's needed to sort the list with the unique classes to produce the plots in the right order.
    # If it was guaranteed that the classes are discrete encoded values between 0..n, 
    #   this sort would be unnecessary (could handle it pretty easily using np.arange).
    # Considering this isn't guaranteed at all (there isn't a built-in encoder in BioAutoML),
    #   the redundant solution will be kept as a backup for any dataset with words as classes.
    classes = sorted(set(targets))
    assert len(shap_values) == len(classes),\
        "Error: Classes generated by the explainer of 'model' doesn't match the distinct number " +\
        f"of classes in 'targets'. [Explainer={len(shap_values)}, Target={len(classes)}]"

    if not os.path.exists(path):
        print(f"Creating explanations directory: {path}...")
        os.mkdir(path)
    else:
        print(f"Directory {path} already exists. Will proceed using it...")

    generated_plt[PLOT_NAME_SUMMARY] = []
    generated_plt[PLOT_NAME_WATERFALL] = []

    print("Plotting each class with summary and waterfall plots...")
    for i, cl in enumerate(classes):
        generated_plt[PLOT_NAME_SUMMARY].append(
            generate_summary_plot(shap_values[i], cl, features, feature_names, path)
        )

        random_samples = randomize_samples(targets, cl, n_samples=n_samples)
        base_value = explainer.expected_value[i]
        for j, sample in enumerate(random_samples):
            generated_plt[PLOT_NAME_WATERFALL].append(
                generate_waterfall_plot(j+1, shap_values[i][sample], cl, features[sample], 
                                        base_value, feature_names, path)
            )

    return generated_plt


def build_interpretability_report(generated_plt,  n_samples, report_name="interpretability.pdf", directory="."):
    report = Report(report_name, directory=directory)
    
    root_dir = os.path.abspath(os.path.join(__file__, os.pardir))
    report.insert_doc_header(REPORT_MAIN_TITLE_MULTICLASS, logo_fig=os.path.join(root_dir, "img/BioAutoML.png"))
    report.insert_text_on_doc(REPORT_SHAP_PREAMBLE, font_size=12, pos_margin=1)

    report.insert_text_on_doc(REPORT_SUMMARY_TITLE, font_size=14, style="Center", pre_margin=18, pos_margin=12, bold=True)
    report.insert_figure_on_doc(generated_plt[PLOT_NAME_SUMMARY])
    report.insert_text_on_doc(REPORT_SHAP_SUMMARY_1, font_size=12)
    report.insert_text_on_doc(REPORT_SHAP_SUMMARY_2, font_size=12, pos_margin=1)

    report.insert_text_on_doc(REPORT_WATERFALL_TITLE, font_size=14, style="Center", pre_margin=18, pos_margin=12, bold=True)
    report.insert_text_on_doc(REPORT_SHAP_WATERFALL_1(n_samples), font_size=12)
    report.insert_text_on_doc(REPORT_SHAP_WATERFALL_2, font_size=12)
    report.insert_figure_on_doc(generated_plt[PLOT_NAME_WATERFALL])

    report.build()


def multiclass_pipeline(test, test_labels, test_nameseq, norm, classifier, tuning, output, exp_n_samples):

    global clf, train, train_labels

    if not os.path.exists(output):
        os.mkdir(output)

    train = train_read
    train_labels = train_labels_read
    column_train = train.columns
    column_test = ''

    #  tmp = sys.stdout
    #  log_file = open(output + 'task.log', 'a')
    #  sys.stdout = log_file

    """Number of Samples and Features: Train and Test"""

    print('Number of samples (train): ' + str(len(train)))

    """Number of labels"""

    print('Number of Labels (train):')
    df_label = pd.DataFrame(train_labels)
    print(str(pd.value_counts(df_label.values.flatten())))

    if os.path.exists(ftest) is True:
        column_test = test.columns
        print('Number of samples (test): ' + str(len(test)))
        print('Number of Labels (test):')
        df_label = pd.DataFrame(test_labels)
        print(str(pd.value_counts(df_label.values.flatten())))

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
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        train = pd.DataFrame(imp.fit_transform(train), columns=column_train)
        if os.path.exists(ftest) is True:
            test = pd.DataFrame(imp.transform(test), columns=column_test)
    else:
        print('There are no missing values...')

    """Preprocessing: StandardScaler()"""

    if norm is True:
        print('Applying StandardScaler()....')
        sc = StandardScaler()
        train = pd.DataFrame(sc.fit_transform(train), columns=column_train)
        if os.path.exists(ftest) is True:
            test = pd.DataFrame(sc.transform(test), columns=column_test)

    """Choosing Classifier """

    print('Choosing Classifier...')
    if classifier == 0:
        print('Tuning: ' + str(tuning))
        print('Classifier: XGBClassifier')
        clf = xgb.XGBClassifier(eval_metric='mlogloss', random_state=63, use_label_encoder=False)
        # train, train_labels = imbalanced_function(clf, train, train_labels)
        if tuning is True:
            print('Tuning not yet available for XGBClassifier')
    elif classifier == 1:
        print('Tuning: ' + str(tuning))
        print('Classifier: Random Forest')
        clf = RandomForestClassifier(n_estimators=200, n_jobs=n_cpu, random_state=63)
        # train, train_labels = imbalanced_function(clf, train, train_labels)
        if tuning is True:
            best_tuning, clf = tuning_rf_bayesian()
            print('Finished Tuning')
    elif classifier == 2:
        print('Tuning: ' + str(tuning))
        print('Classifier: LightGBM')
        clf = lgb.LGBMClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)
        # train, train_labels = imbalanced_function(clf, train, train_labels)
        if tuning is True:
            best_tuning, clf = tuning_lightgbm_bayesian()
            print('Finished Tuning')
    else:
        sys.exit('This classifier option does not exist - Try again')

    """Preprocessing: Feature Importance-Based Feature Selection"""

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

    print('Best Feature Subset: ' + str(len(feature_name)))
    print('Reduction: ' + str(len(column_train)-len(feature_name)) + ' features')
    fs_train = os.path.join(output, 'best_feature_train.csv')
    fs_test = os.path.join(output, 'best_feature_test.csv')
    print('Saving dataset with selected feature subset - train: ' + fs_train)
    train.to_csv(fs_train, index=False)
    if os.path.exists(ftest) is True:
        print('Saving dataset with selected feature subset - test: ' + fs_test)
        test.to_csv(fs_test, index=False)
    print('Feature Selection - Finished...')

    """Training - StratifiedKFold (cross-validation = 10)..."""

    print('Training: StratifiedKFold (cross-validation = 10)...')
    train_output = os.path.join(output, 'training_kfold(10)_metrics.csv')
    matrix_output = os.path.join(output, 'training_confusion_matrix.csv')
    model_output = os.path.join(output, 'trained_model.sav')
    evaluate_model_cross(train, train_labels, clf, train_output, matrix_output)
    clf.fit(train, train_labels)
    joblib.dump(clf, model_output)
    print('Saving results in ' + train_output + '...')
    print('Saving confusion matrix in ' + matrix_output + '...')
    print('Saving trained model in ' + model_output + '...')
    print('Training: Finished...')

    """Generating Feature Importance - Selected feature subset..."""

    print('Generating Feature Importance - Selected feature subset...')
    importance_output = os.path.join(output, 'feature_importance.csv')
    features_importance_ensembles(clf, feature_name, importance_output)
    print('Saving results in ' + importance_output + '...')

    """Testing model..."""

    if os.path.exists(ftest) is True:
        print('Generating Performance Test...')
        preds = clf.predict(test)
        pred_output = os.path.join(output, 'test_predictions.csv')
        print('Saving prediction in ' + pred_output + '...')
        save_prediction(preds, test_nameseq, pred_output)

        """Generating Explainable Machine Learning plots from the test set..."""

        try:
            plot_output = os.path.join(output, 'explanations')
            generated_plt = generate_all_plots(clf, test.values, test.columns, preds,
                                                path=plot_output, n_samples=exp_n_samples)
            build_interpretability_report(generated_plt, exp_n_samples, directory=output)
        except ValueError as e:
            print(e)
            print("If you believe this is a bug, please report it to https://github.com/Bonidia/BioAutoML.")
            print("Generation of explanation plots and report failed. Proceeding without it...")
        except AssertionError as e:
            print(e)
            print("This is certainly a bug. Please report it to https://github.com/Bonidia/BioAutoML.")
            print("Generation of explanation plots and report failed. Proceeding without it...")
        else:
            print("Explanation plots and report generated successfully!")

        if os.path.exists(ftest_labels) is True:
            print('Generating Metrics - Test set...')
            report = classification_report(test_labels, preds, output_dict=True)
            matrix_test = (pd.crosstab(test_labels, preds, rownames=["REAL"], colnames=["PREDITO"], margins=True))

            metrics_output = os.path.join(output, 'metrics_test.csv')
            print('Saving Metrics - Test set: ' + metrics_output + '...')
            metr_report = pd.DataFrame(report).transpose()
            metr_report.to_csv(metrics_output)

            matrix_output_test = os.path.join(output, 'test_confusion_matrix.csv')
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


##########################################################################
##########################################################################
if __name__ == '__main__':
    print('\n')
    print('###################################################################################')
    print('###################################################################################')
    print('#####################        BioAutoML - MultiClass         #######################')
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
    parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
    parser.add_argument('-classifier', '--classifier', default=0,
                        help='Classifier - 0: XGBoost, 1: Random Forest '
                             '2: LightGBM')
    parser.add_argument('-tuning', '--tuning_classifier', type=bool, default=False,
                        help='Tuning Classifier - True = Yes, False = No, default = False')
    parser.add_argument('-output', '--output', help='results directory, e.g., result/')
    parser.add_argument('-n_exp_samples', '--n_exp_samples', default=3, 
                        help='number of samples taken for each class in explanation analysis')
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
    n_exp_samples = int(args.n_exp_samples)
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

    multiclass_pipeline(
        test_read, test_labels_read, test_nameseq_read, norm, classifier, 
        tuning, foutput, n_exp_samples
    )
    cost = (time.time() - start_time)/60
    print('Computation time - Pipeline: %s minutes' % cost)
##########################################################################
##########################################################################
