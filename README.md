# POS Tagging: Viterbi Algorithm and Deep Learning Comparision

This project contains two python implementation of Viterbi Algorithm on Hidden Markov Models for POS tagging:
1. Bigram Viterbi Algorithm
2. Trigram Viterbi Algorithm

The report contains results comparision of both implementation on the:
1. English Penn treebank dataset
2. Japanese language dataset
3. Bulgarian language dataset

I have also implemented Bidirectional LSTM with Word Embeddings as initial weights for English Penn treebank dataset and analyzed the performance (word error rate and sentence error rate) against Viterbi Algorithm

### Dataset analysis 
![analysis](https://github.com/sankalpapharande/POS_Tagging/blob/main/pos_tagging/result_plots/Screen%20Shot%202022-09-16%20at%203.36.57%20PM.png)


Results on the output of the Viterbi python algorithm
1. Bigram HMM and Bigram Viterbi:
error rate by word: 0.05351845850886158 (2147 errors out of 40117)
error rate by sentence: 0.6470588235294118 (1100 errors out of 1700)

2. trigram HMM and Trigram Viterbi
error rate by word: 0.05005359323977366 (2008 errors out of 40117)
error rate by sentence: 0.6317647058823529 (1074 errors out of 1700)

# Result Analysis:
### Bigram vs Trigam Viterbi HMM on English Penn Treebank dataset
![Screen Shot 2022-09-16 at 3.36.01 PM.png](https://github.com/sankalpapharande/POS_Tagging/blob/main/pos_tagging/result_plots/Screen%20Shot%202022-09-16%20at%203.36.01%20PM.png)


### Model Learning Curves
![Learning curves](https://github.com/sankalpapharande/POS_Tagging/blob/main/pos_tagging/result_plots/Screen%20Shot%202022-09-16%20at%203.36.30%20PM.png)

### Bigram vs Trigram for 
1. English Penn treebank dataset
2. Japanese language dataset
3. Bulgarian language dataset

![all_language](https://github.com/sankalpapharande/POS_Tagging/blob/main/pos_tagging/result_plots/Screen%20Shot%202022-09-16%20at%203.36.43%20PM.png)

### Comparision
![comparision](https://github.com/sankalpapharande/POS_Tagging/blob/main/pos_tagging/result_plots/Screen%20Shot%202022-09-16%20at%203.37.07%20PM.png)

### Bigram vs Trigram vs Bidirectional LSTM for 3 laguages:
![all](https://github.com/sankalpapharande/POS_Tagging/blob/main/pos_tagging/result_plots/Screen%20Shot%202022-09-16%20at%203.37.21%20PM.png)
