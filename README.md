# POS Tagging: Viterbi Algorithm and Deep Learning Comparision

This project contains two python implementation of Viterbi Algorithm on Hidden Markov Models for POS tagging:
1. Bigram Viterbi Algorithm
2. Trigram Viterbi Algorithm

The report contains results comparision of both implementation on the:
1. English Penn treebank dataset
2. Japanese language dataset
3. Bulgarian language dataset

I have also implemented Bidirectional LSTM with Word Embeddings as initial weights for English Penn treebank dataset and analyzed the performance (word error rate and sentence error rate) against Viterbi Algorithm

Results on the output of the Viterbi python algorithm
1. Bigram HMM and Bigram Viterbi:
error rate by word: 0.05351845850886158 (2147 errors out of 40117)
error rate by sentence: 0.6470588235294118 (1100 errors out of 1700)

2. trigram HMM and Trigram Viterbi
error rate by word: 0.05005359323977366 (2008 errors out of 40117)
error rate by sentence: 0.6317647058823529 (1074 errors out of 1700)

