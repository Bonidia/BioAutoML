![Python](https://img.shields.io/badge/python-v3.7-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-up-brightgreen)

<h1 align="center">
</h1>

<h4 align="center">BioAutoML: Automated Feature Engineering and Metalearning for Classification of Biological Sequences</h4>

<p align="center">
  <a href="https://bonidia.github.io/BioAutoML/">Home</a> •
  <a href="https://github.com/Bonidia/BioAutoML/">Repository</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a> 
</p>

<h1 align="center"></h1>

## BioAutoML - Metalearning - Binary Problems - Using features from other packages

To use this model, follow the example below:

```sh 
To run the code (Example): $ python BioAutoML-binary.py -h


Where:

-train: csv format file, e.g., train.csv

-train_label: csv format file with labels, e.g., labels_test.csv

-test: csv format file, e.g., test.csv

-test_label: csv format file with labels, e.g., labels_test.csv

-test_nameseq: csv with sequence names - test

-nf: Normalization - Features (default = False)

-n_cpu: number of cpus - default = 1
  
-classifier: Classifier - 0: CatBoost, 1: Random Forest 2: LightGBM
  
-imbalance: To deal with the imbalanced dataset problem - True = Yes, False = No, default = False

-tuning: Tuning Classifier - True = Yes, False = No, default = False

-output: results directory, e.g., result
```

**Running:**

```sh
$ python BioAutoML-binary.py -train example_csv/lncRNA/train-human.csv -train_label example_csv/lncRNA/train-human-labels.csv -test example_csv/lncRNA/test-human.csv -test_label example_csv/lncRNA/test-human-labels.csv -test_nameseq example_csv/lncRNA/test-human-sequences.csv -classifier 2 -output example_results

or

$ python BioAutoML-binary.py -train example_csv/lncRNA/train-human.csv -train_label example_csv/lncRNA/train-human-labels.csv -test example_csv/lncRNA/test-human.csv -test_label example_csv/lncRNA/test-human-labels.csv -test_nameseq example_csv/lncRNA/test-human-sequences.csv -imbalance True -tuning True -classifier 2 -output example_results
```

**Note** This example is in the Directory: example_csv/lncRNA
**Note** Inserting a test dataset is optional. 
