# GDLC: Graph Deep Learning framework based on Centrality measures for IDS

## Overview
GDLC (Graph Deep Learning based on Centrality measures) is a novel framework designed to enhance Network Intrusion Detection Systems (NIDS). By leveraging centrality measures within graph-based deep learning models, GDLC aims to improve the detection rates and accuracy of identifying network intrusions, addressing the dynamic and sophisticated nature of cyber threats in IoT environments.

Implementation using Pytorch Lightning and Weight and Baises.

## Project Structure
```
├── main.py                   # Main file for running experiments
├── pre_processing.ipynb      # Data preprocessing steps
├── prepare_gdlc_files.ipynb  # Graph preparation scripts
├── requirements.txt          # Dependencies required for the project
├── src/                      # Source code directory
```

## Usage
1. **Preprocess the Data**:
   Run `pre_processing.ipynb` to clean and prepare the dataset.

2. **Prepare GDLC Files**:
   Execute `prepare_gdlc_files.ipynb` to generate gdlc structures.

3. **Run Experiments**:
   Use `main.py` to train and evaluate GDLC models.

## Datasets
The project supports multiple NIDS datasets, including:
- CIC-TON IoT
- CIC-IDS 2017
- CIC-Bot IoT
- CCD-INID
- NF-UQ-NIDS
- Edge-IIoT
- NF-CSE-CIC-IDS2018
- X-IIoT

## Contributions
Feel free to contribute by submitting issues or pull requests.

## Contact
For questions or collaborations, reach out to the author at [mtermos@cesi.fr].