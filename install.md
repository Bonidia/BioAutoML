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
