# MutStab-ESM
## About

In modern molecular biology, understanding the impact of missense mutations on protein stability and function is of utmost importance. Experimental determination of the effects of mutations on stability is often costly, highlighting the urgent need to develop accurate mutation prediction methods. In recent years, prediction methods have become valuable resources for assessing the impact of mutations on protein stability. Missense mutations can be categorized into two types based on their effects on protein stability: destabilizing mutations and stabilizing mutations. In the current research on protein stability prediction methods, predicting stabilizing mutations remains a major challenge. Therefore, we have developed a machine learning prediction method called MutStab-ESM, which focuses on the accurate prediction of stabilizing mutations.

We have introduced an innovative data augmentation strategy that involves generating all potential mutations for a protein and integrating the results of various prediction methods. Through this approach, we successfully constructed a high-quality dataset comprising 482,222 mutations. Using this dataset, our invention leverages transfer learning techniques to extract embedding features from the large protein model ESM-2. Our invention employs a Transformer Encoder-based architecture, focusing on analyzing the information changes in amino acid residues at mutation sites before and after the mutation, thereby predicting the mutation's impact on protein stability.

Our method was compared with other approaches across multiple independent test sets, and the results demonstrated that our method not only exhibits higher reliability and stability but also achieves prediction performance comparable to structure-based models by solely using protein sequence information. This method holds promise for providing guidance in the fields of medicine and protein engineering, not only aiding in the identification of harmful mutations for early diagnosis and intervention but also offering direction for drug development and advancing therapeutic solutions.



## Installation

1. Python packages: Pytorch

```
pip install torch
```



## Running

1. Environment

We provide the Python environment for running the program, with the installation as follows:

```
mamba env create -f pyG.yml
```


2. Prepare

Prepare a FASTA-formatted sequence file that includes both the wild-type and mutant sequences.

![image](https://github.com/user-attachments/assets/d77ad444-3951-48aa-bf78-5e1827e8c6ae)


3. Calculate ESM embedding

```
python ~/MutStab-ESM/ESM/esm/scripts/extract.py  esm2_t36_3B_UR50D  Q14232.fa output --repr_layers -1 --include mean per_tok --truncation_seq_length 2000

```


4. Predict

```
python /public/home/zff/MutStab-ESM/run_esm_prediction.py -s /public/home/zff/MutStab-ESM/example/Q14232.fa -m K11A -f /public/home/zff/MutStab-ESM/example/output/ -o /public/home/zff/MutStab-ESM/example/
```

-s: Sequence file

-m: Mutation

-f: Path of the Protein embedding files

-o: Output path
