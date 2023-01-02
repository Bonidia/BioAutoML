![Python](https://img.shields.io/badge/python-v3.7-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-up-brightgreen)

<h1 align="center">
</h1>

<h4 align="center">BioAutoML: Automated Feature Engineering and Metalearning for Classification of Biological Sequences</h4>

<h4 align="center">Democratizing Machine Learning in Life Sciences</h4>

<p align="center">
  <a href="https://bonidia.github.io/BioAutoML/">Home</a> •
  <a href="https://github.com/Bonidia/BioAutoML/">Repository</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a> 
</p>

<h1 align="center"></h1>

## Awards

⭐ Finalist Project (Top 15 of 82), Falling Walls Lab Brazil 2022, DWIH São Paulo, Falling Walls Foundation, DAAD The German Center for Science and Innovation [[Link](https://www.youtube.com/watch?v=H5C_UIgVeQM)].

⭐ Latin America Research Awards (LARA), Google, 2021. Project: BioAutoML: Automated Feature Engineering for Classification of Biological Sequences. Elected by LARA-Google among the 24 most promising ideas in Latin America - 2021 - [[Link](https://blog.google/intl/pt-br/novidades/iniciativas/conheca-os-vencedores-do-premio-lara-2021-o-programa-de-bolsas-de-pesquisa-do-google/)] [[Link](http://www.saocarlos.usp.br/programa-de-bolsas-do-google-premia-trabalhos-orientados-pelo-cemeai/)].

## Abstract

Recent technological advances allowed an exponential expansion of biological sequence data and the extraction of meaningful information through Machine Learning (ML) algorithms. This knowledge improved the understanding of the mechanisms related to several fatal diseases, e.g., Cancer and COVID-19, helping to develop innovative solutions, such as CRISPR-based gene editing, coronavirus vaccine, and precision medicine. These advances benefit our society and economy, directly impacting people's lives in various areas, such as health care, drug discovery, forensic analysis, and food processing. Nevertheless, ML-based approaches to biological data require representative, quantitative, and informative features. Many ML algorithms can handle only numerical data, so sequences need to be translated into a numerical feature vector. This process, known as feature extraction, is a fundamental step for elaborating high-quality ML-based models in bioinformatics, by allowing the feature engineering stage, with the design and selection of suitable features. Feature engineering, ML algorithm selection, and hyperparameter tuning are often manual and time-consuming processes, requiring extensive domain knowledge. To deal with this problem, we present a new package, BioAutoML. BioAutoML automatically runs an end-to-end ML pipeline, extracting numerical and informative features from biological sequence databases, using the MathFeature package, and automating the feature selection, ML algorithm(s) recommendation and tuning of the selected algorithm(s) hyperparameters, using Automated ML (AutoML). BioAutoML has two components, divided into four modules, (1) automated feature engineering (feature extraction and selection modules) and (2) Metalearning (algorithm recommendation and hyper-parameter tuning modules). We experimentally evaluate BioAutoML in two different scenarios: (i) prediction of the three main classes of ncRNAs and (ii) prediction of the seven categories of ncRNAs in bacteria, including housekeeping and regulatory types. To assess BioAutoML predictive performance, it is experimentally compared with two other AutoML tools (RECIPE and TPOT). According to the experimental results, BioAutoML can accelerate new studies, reducing the cost of feature engineering processing and either keeping or improving predictive performance.

* First study to propose an automated feature engineering and metalearning pipeline for ncRNA sequences in bacteria;
    
* BioAutoML can be applied in multi-class and binary problems;
    
* BioAutoML can be used in other DNA/RNA sequences scenarios;
    
* BioAutoML can accelerate new studies, reducing the feature engineering time-consuming stage and improving the design and performance of ML pipelines in bioinformatics;
    
* BioAutoML does not require specialist human assistance.


## Authors

* Robson Parmezan Bonidia, Anderson Paulo Avila Santos, Breno Lívio Silva de Almeida, Peter F. Stadler, Ulisses Nunes da Rocha, Danilo Sipoli Sanches, and André Carlos Ponce de Leon Ferreira de Carvalho.

* **Correspondence:** rpbonidia@gmail.com or bonidia@usp.br

## Documentation

- [Installing dependencies and package](install.md)
- [BioAutoML - Automated Feature Engineering and Metalearning - DNA/RNA](bio.md)
- [BioAutoML - Automated Feature Engineering and Metalearning - With Numerical Mapping - DNA/RNA](bio-mapping.md)
- [BioAutoML - Automated Feature Engineering and Metalearning - Protein](protein.md)
- [BioAutoML - Automated Feature Engineering and Metalearning - Protein - BioAutoML+iFeature](protein-v2)
- [BioAutoML - Metalearning - Binary Problems - Using features from other packages](binary.md)
- [BioAutoML - Metalearning - Multiclass Problems - Using features from other packages](multi.md)

## Citation

If you use this code in a scientific publication, we would appreciate citations to the following paper:

Robson P Bonidia, Anderson P Avila Santos, Breno L S de Almeida, Peter F Stadler, Ulisses N da Rocha, Danilo S Sanches, André C P L F de Carvalho, BioAutoML: automated feature engineering and metalearning to predict noncoding RNAs in bacteria, Briefings in Bioinformatics, 2022, bbac218, [[DOI](https://doi.org/10.1093/bib/bbac218)].

```sh

@article{10.1093/bib/bbac218,
    author = {Bonidia, Robson P and Santos, Anderson P Avila and de Almeida, Breno L S and Stadler, Peter F and da Rocha, Ulisses N and Sanches, Danilo S and de Carvalho, André C P L F},
    title = "{BioAutoML: automated feature engineering and metalearning to predict noncoding RNAs in bacteria}",
    journal = {Briefings in Bioinformatics},
    year = {2022},
    month = {06},
    issn = {1477-4054},
    doi = {10.1093/bib/bbac218},
    url = {https://doi.org/10.1093/bib/bbac218},
    note = {bbac218},
}

```
