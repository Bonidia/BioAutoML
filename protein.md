
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

## BioAutoML - Automated Feature Engineering and Metalearning - End-to-end Machine Learning Workflow - Protein

To use this model, follow the example below:

```sh 
To run the code (Example): $ python BioAutoML-feature-protein.py -h


Where:

-fasta_train: fasta format file, e.g., fasta/lncRNA.fasta fasta/circRNA.fasta
 
-fasta_label_train: labels for fasta files, e.g., lncRNA circRNA

-fasta_test: fasta format file, e.g., fasta/lncRNA.fasta fasta/circRNA.fasta

-fasta_label_test: labels for fasta files, e.g., lncRNA circRNA

-estimations: number of estimations - BioAutoML - default = 50

-n_cpu: number of cpus - default = 1

-output: results directory, e.g., result
```

**Running:**

```sh
$ python BioAutoML-feature-protein.py -fasta_train MathFeature/Case\ Studies/CS-I/train_P.fasta MathFeature/Case\ Studies/CS-I/train_N.fasta -fasta_label_train positive negative -fasta_test MathFeature/Case\ Studies/CS-I/test_P.fasta MathFeature/Case\ Studies/CS-I/test_N.fasta -fasta_label_test positive negative -output experimental/protein
```

**Note** This example is in the Directory: MathFeature. 
