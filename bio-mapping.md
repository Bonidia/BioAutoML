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

## BioAutoML - Automated Feature Engineering and Metalearning - With Numerical Mapping - End-to-end Machine Learning Workflow

To use this model, follow the example below:

```sh 
To run the code (Example): $ python BioAutoML-feature-mapping.py -h


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
$ python BioAutoML-feature-mapping.py -fasta_train Case\ Studies/CS-I-A/E_coli/train/rRNA.fasta Case\ Studies/CS-I-A/E_coli/train/sRNA.fasta -fasta_label_train rRNA sRNA -fasta_test Case\ Studies/CS-I-A/E_coli/test/rRNA.fasta Case\ Studies/CS-I-A/E_coli/test/sRNA.fasta -fasta_label_test rRNA sRNA -output test_directory
```

**Note** This example is in the Directory: Case Studies. 

**Note** Inserting a test dataset is optional. 


**Running: In unknown sequences**

```sh
$ python BioAutoML-feature.py -fasta_train Case\ Studies/CS-I-A/E_coli/train/rRNA.fasta Case\ Studies/CS-I-A/E_coli/train/sRNA.fasta -fasta_label_train rRNA sRNA -fasta_test new_sequences.fasta -fasta_label_test unknown -output test_directory
```


