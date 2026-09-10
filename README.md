# UltraBench

[![arXiv](https://img.shields.io/badge/arXiv-2501.13193-b31b1b.svg)](http://arxiv.org/abs/2501.13193)
[![OpenReview](https://img.shields.io/badge/OpenReview-Reviews-blue.svg)](https://openreview.net/forum?id=iGcxlTLIL5)

UltraBench is a standardized benchmark of 21 convex ultrasound image classification and semantic segmentation tasks drawn from 12 publicly available datasets covering 11 regions of the body. This repository contains the datasets, the downstream training framework, and the evaluation layer used to train and score machine learning models on them.

This benchmark was published alongside the article [Revisiting Data Augmentation for Ultrasound Images](https://openreview.net/forum?id=iGcxlTLIL5). If you use this benchmark in your research please cite this article and the articles for each of the included datasets. Instructions for how to do so are provided at the end.

## Motivation

A common limitation of studies on ultrasound image analysis using machine learning is a lack of evaluations across multiple domains. However, this is made difficult by a lack of ultrasound image analysis tasks in existing medical image analysis benchmarks, such as [MedMNIST](https://www.nature.com/articles/s41597-022-01721-8), [MedSegBench](https://www.nature.com/articles/s41597-024-04159-2) and the [Medical Segmentation Decathlon](https://www.nature.com/articles/s41467-022-30695-9). To address this issue, we created UltraBench. Our aim is to make it easier for researchers by standardizing the preprocessing of many publicly available ultrasound datasets and providing tools (e.g., dataset implementations for common deep learning frameworks) that (a) reduce the effort required to use them and (b) make it easier for researchers to compare results between papers.

## Installation

Install the latest version of UltraBench from PyPI using:

```bash
pip install ultrabench
```

See [Contributing](#contributing) for instructions on setting up a development environment.

## Getting Started

Load any dataset in the collection with one function:

```python
import ultrabench

train_data = ultrabench.load_dataset(name="AUL", split="train", revision="0.4")
```

Most datasets are published preprocessed and download ready to use. A few cannot be republished in that form, so `load_dataset` preprocesses them on your machine the first time and caches the result. Two of those cannot be republished at all, so you download the raw data yourself and pass its directory:

```python
train_data = ultrabench.load_dataset(name="GBCU", path="path/to/raw/data")
```

See [the dataset documentation](docs/dataset_format.md) for the three kinds of dataset, where the cache lives, and how to pin a revision.

UltraBench is also packaged as a commandline application, for preprocessing raw datasets and publishing them. After installing, run `ultrabench --help` to get started.

For more information, checkout the [Documentation](https://github.com/adamtupper/ultrabench/wiki) on the wiki!

## Current Datasets & Tasks

The following table contains the list of supported datasets and tasks. Click on the article and data links to visit the original publications.

The **Access** column says how each dataset reaches you. *Processed* datasets are published ready to use. *Raw* datasets may be redistributed unmodified but not preprocessed, so `load_dataset` downloads the original files and preprocesses them on your machine. *Local* datasets may not be redistributed at all, so you download them from the original source and pass the directory as `path`.

| Dataset | Access | Region | Classification Tasks | Segmentation Tasks |
|---------|--------|--------|----------------------|--------------------|
| Annotated Ultrasound Liver ([article](https://doi.org/10.1093/bib/bbac569), [data](https://zenodo.org/records/7272660)) | Processed | Liver | Liver mass classification | Liver segmentation, Liver mass segmentation |
| BUS-BRA ([article](https://doi.org/10.1002/mp.16812), [data](https://zenodo.org/records/8231412)) | Processed | Breast | Histology classification, Pathology classification | Breast lesion segmentation |
| CAMUS ([article](https://ieeexplore.ieee.org/document/8649738/), [data](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)) | Processed | Heart | Image quality classification | Cardiac region segmentation |
| COVID-BLUES ([article](https://doi.org/10.1109/TUFFC.2025.3535879), [data](https://github.com/NinaWie/COVID-BLUES)) | Raw | Lungs | COVID-19 classification | |
| Fetal Planes ([article](https://www.nature.com/articles/s41598-020-67076-5), [data](https://zenodo.org/records/3904280)) | Processed | Fetus | Plane classification, Brain plane classification | |
| Gallbladder Cancer Ultrasound ([article](https://ieeexplore.ieee.org/document/9879895), [data](https://gbc-iitd.github.io/data/gbcu)) | Local | Gallbladder | Gallbladder tumour classification | |
| MASLD ([article](https://link.springer.com/article/10.1007/s11548-018-1843-2), [data](https://zenodo.org/records/1009146)) | Processed | Liver | Steatosis classification | |
| MMOTU ([article](http://arxiv.org/abs/2207.06799), [data](https://github.com/cv516Buaa/MMOTU_DS2Net)) | Raw | Ovaries | Ovarian tumour classification | Ovarian tumour segmentation |
| Open Kidney Ultrasound ([article](https://link.springer.com/chapter/10.1007/978-3-031-44521-7_15), [data](https://rsingla.ca/kidneyUS/)) | Local | Kidney | | Kidney capsule segmentation, Kidney region segmentation |
| PSFHS ([article](https://www.nature.com/articles/s41597-024-03266-4), [data](https://zenodo.org/records/10969427)) | Processed | Fetus | | Fetal head and pubic symphysis segmentation |
| STMUS ([data](https://data.mendeley.com/datasets/3jykz7wz8d)) | Processed | Muscle | | Muscle segmentation |
| TN3K ([article](https://doi.org/10.1016/j.compbiomed.2022.106389), [data](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)) | Processed | Thyroid | Thyroid nodule classification | Thyroid nodule segmentation |

## Contributing

We would love for this benchmark to grow and flourish into a resource that anyone with an interest in machine learning for ultrasound analysis can pick up and use quickly and easily. Any help addressing bugs, contributing new datasets or tasks, or any other improvements are welcome and appreciated! I only ask that you respect the community guidelines laid out in the `CODE_OF_CONDUCT.md`. For more information on how to contribute, checkout out the [Documentation](https://github.com/adamtupper/ultrabench/wiki).

UltraBench is built using [uv](https://docs.astral.sh/uv/). For the best experience, you can use the development container configuration (see [DEVCONTAINER.md](/.devcontainer/DEVCONTAINER.md)) to setup an isolated development environment with uv and all the necessary dependencies already installed. Otherwise, you can [install uv](https://docs.astral.sh/uv/getting-started/installation/) and then following the commands below. To ensure that your code meets the style guidelines etc., this also installs and initializes the optional development dependencies.

```bash
uv sync --group dev --group training  # Install optional dependencies, if required
source .venv/bin/activate
nbstripout --install
pre-commit install
```

## Questions

If you have any questions, please open an issue or contact us via email.

## How to Cite UltraBench

If you use UltraBench in your research, please cite our article [Revisiting Data Augmentation for Ultrasound Images](https://openreview.net/forum?id=iGcxlTLIL5) and the articles for each of the datasets. This ensures that the work of the authors of the original datasets is properly acknowledged, helps more people find the benchmark and encourages us to continue maintaining and improving it!

```
@article{
  tupper2025,
  title={Revisiting Data Augmentation for Ultrasound Images},
  author={Adam Tupper and Christian Gagn{\'e}},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=iGcxlTLIL5},
  note={}
}
```