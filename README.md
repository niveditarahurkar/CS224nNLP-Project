# CS224nNLP-Project
Class project for Stanford CS224n - NLP using Deep Learning.
The project is about Analysing errors occurring in Question-Answering systems using the TyDi dataset (https://arxiv.org/abs/2003.05002)

Steps followed:
1. Trained the mBert base model as a Baseline system.
    To train mbert, we used the code provided by Tydi. We used the gold passage baseline approach. (https://github.com/google-research-datasets/tydiqa/tree/master/gold_passage_baseline)

2. Generated labels for tydi gold passage dev set.

3. Fine tuned mBert on the dev dataset and the label generated to increase the accuracy.(See Python notebook)
