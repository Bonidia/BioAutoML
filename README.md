![Python](https://img.shields.io/badge/python-v3.7-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-up-brightgreen)

<h1 align="center">
  <img src="https://github.com/Bonidia/BioAutoML/blob/main/img/BioAutoML.png" alt="BioAutoML" width="400">
</h1>

<h4 align="center">BioAutoML: Automated Feature Engineering for Classification of Biological Sequences</h4>

<p align="center">
  <a href="https://github.com/Bonidia/MathFeature">Home</a> •
  <a href="https://bonidia.github.io/MathFeature/">Documentation</a> •
  <a href="http://mathfeature.icmc.usp.br/">Web Server</a> •
  <a href="https://github.com/Bonidia/MathFeature-WebServer">Web Server - LocalHost</a> •
  <a href="#list-of-files">List of files</a> •
  <a href="#installing-dependencies-and-package">Installing</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> 
</p>

<h1 align="center"></h1>

## Abstract

Recent technological advances allowed an exponential expansion of biological sequence data, and the extraction of meaningful information through Machine Learning (ML) algorithms. This knowledge improved the understanding of the mechanisms related to several fatal diseases, e.g., Cancer and COVID-19, helping to develop innovative solutions, such as CRISPR-based gene editing, coronavirus vaccine, and precision medicine. These advances benefit our society and economy, directly impacting people’s lives in various areas, such as health care, drug discovery, forensic analysis, and food analysis. Nevertheless, ML approaches applied to biological data require representative, quantitative, and informative features. Necessarily, as many ML algorithms can handle only numerical
data, sequences need to be translated into a feature vector. This process is known as feature extraction, a fundamental step for the elaboration of high-quality ML-based models in bioinformatics, especially in the stage of feature engineering. This process often requires extensive domain knowledge, performed manually by a human expert, making feature engineering a decisive and time-consuming step in the ML pipeline. Thus, we propose to develop a new
package, BioAutoML, able to extract relevant numerical information from biological sequences. BioAutoML will use Automated ML (AutoML) to recommend the best feature vector to be extracted from a biological dataset. Fundamentally, this project is divided into two stages: (1) Implement feature extraction descriptors for biological sequences, and (2) automate efficient and robust feature extraction pipelines. The first experimental results, assessing the relevance of the implemented descriptors, indicate robust results for different problem domains, such as SARS-CoV-2, anticancer peptides, HIV sequences, and non-coding RNAs. According to our systematic review, our proposal is innovative compared to available studies in the literature, being the first study to propose automated feature engineering for biological sequences, allowing non-experts to use relevant feature extraction techniques.

## Authors

* Robson Parmezan Bonidia, Danilo Sipoli Sanches, and André Carlos Ponce de Leon Ferreira de Carvalho.

* **Correspondence:** rpbonidia@gmail.com or bonidia@usp.br


## Publication

Robson P Bonidia, Douglas S Domingues, Danilo S Sanches, André C P L F de Carvalho, MathFeature: feature extraction package for DNA, RNA and protein sequences based on mathematical descriptors, Briefings in Bioinformatics, 2021; bbab434, https://doi.org/10.1093/bib/bbab434.

## List of files

 - **case studies:** case studies used in our article;
 - **GUI:** GUI (Graphical User Interface)-based platform;
 - **examples:** Files of Example;
 - **files:** files used in some methods;
 - **methods:** Main Files - Feature Extraction Models, e.g., Fourier, Numerical Mapping, Entropy, Complex Networks;
 - **preprocessing:** Preprocessing Files;
 - **README:** Documentation;
 - **requirements:** Dependencies.


## Installing dependencies and package

## Conda - Terminal

Installing BioAutoML using miniconda, e.g.:

```sh
$ git clone https://github.com/Bonidia/BioAutoML.git BioAutoML

$ cd BioAutoML

$ git submodule init

$ git submodule update
```

**1 - Install Miniconda:** 

```sh

See documentation: https://docs.conda.io/en/latest/miniconda.html

$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$ chmod +x Miniconda3-latest-Linux-x86_64.sh

$ ./Miniconda3-latest-Linux-x86_64.sh

$ export PATH=~/miniconda3/bin:$PATH

```

**2 - Create environment:**

```sh

conda env create -f BioAutoML-env.yml -n bioautoml

```

**3 - Activate environment:**

```sh

conda activate bioautoml

```

**4 - You can deactivate the environment, using:**

```sh

conda deactivate

```
## How to use

See our [documentation](https://bonidia.github.io/MathFeature).

## Citation

If you use this code in a scientific publication, we would appreciate citations to the following paper:

Robson P Bonidia, Douglas S Domingues, Danilo S Sanches, André C P L F de Carvalho, MathFeature: feature extraction package for DNA, RNA and protein sequences based on mathematical descriptors, Briefings in Bioinformatics, 2021; bbab434, https://doi.org/10.1093/bib/bbab434.

```sh

@article{10.1093/bib/bbab434,
    author = {Bonidia, Robson P and Domingues, Douglas S and Sanches, Danilo S and de Carvalho, André C P L F},
    title = "{MathFeature: feature extraction package for DNA, RNA and protein sequences based on mathematical descriptors}",
    journal = {Briefings in Bioinformatics},
    year = {2021},
    month = {11},
    issn = {1477-4054},
    doi = {10.1093/bib/bbab434},
    url = {https://doi.org/10.1093/bib/bbab434},
    note = {bbab434},
    eprint = {https://academic.oup.com/bib/advance-article-pdf/doi/10.1093/bib/bbab434/41108442/bbab434.pdf},
}

```
