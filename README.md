# Query2Cohort

Interpreting Natural Language Clinical Queries

Note: Due to being a consulting project, data and other information is under NDA. This repository contains only the notebooks demonstrating model performance as well as scripts required to train models on own data. Please refer to Excluded Code subtitle for more information to what has been excluded for now. 

**Company**: [Omic](Omic.ai)

## Background

Query2Cohort is part of a consulting project as part of Insight for Omic. Cohort identification and formation for clinical research is a notoriously time and effort consuming process. Some of this burden can be alleviated by using query languages to filter against data to return cohorts matching certain inclusion and exclusion criterias. There exists industry standard tools and software that can assist in the automation of these queries (http://www.ohdsi.org/web/criteria2query/). However, these tools are based primarily on rule-based heuristic models and require user interaction at some stages to utilize. While these tools facilitate the cohort identification process and can query against the company's own data, limitations arise from cross compatibility, requiring multiple software to get from query to results, as well as still requiring user intervention. 

Deep learning has shown promise in cohort identification but is still in its infancy stages with a lack of end-to-end pipelines to achieve data supported answers from a simple enter of user query. Implementation of deep learning models and design provides advantages over traditional methods by allowing for better generalization to unseen formats, less time to adapt to future unseen query types, and interconnectivity with other modules/tools within an OS through transfer of embeddings. 

## Tasks

There are 3 tasks that must be performed on user query written in English: criteria identification, intent analysis, and entity standardization. The first two tasks are utilize BERT to achieve high accuracy while entity normalization utilizes word embeddings with Word Mover's Distance for conversion.


## Data

To use the scripts to train on user's data, should follow the following format:

CSV with 2 columns - query and cohort. Cohort column should be structured as dictionary with two keys - inclusion and exclusion. Values for these keys must be lists.
    - e.g. {"inclusion": ["example a", "example b"], "exclusion": ["exclusion a"]}
    
 To use scripts for inference, data can be simply user queries written in English. 


## Training

To use the scripts for training, users can call the script followed by the path to the file followed by number of training epochs.

```
criteria_identification_training.py data.csv 6
```
## Inference

To use the scripts for inference, user can pass in a csv with one column titled "queries" as a secondary command line argument.

```
criteria_identification_inference.py data.csv
```

## Excluded Code
Anticipated time for NDA update is week of 7-13-20.

Due to NDA, notebook/script for entity normalization algorithm with BioWordVec embeddings and WMD against ICD concepts will not be publically uploaded at present. On full release of Query2Cohort, this code will be pushed. In the mean time, please reach out to the provided contact information if you wish to discuss this novel algorithm.

Due to NDA, performance on test/production data as well as associated code/notebooks are not at liberty to be disclosed. However, results match with expectations from the validation set and any questions regarding general performance can be discussed in private through the following contact information.

## Contact

For questions or comments, feel free to reach out to kevin.e.yie@gmail.com. Thank you!
