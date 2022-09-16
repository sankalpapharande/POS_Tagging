import os


def analyze_japanese_bi_gram():
    os.system("python train_hmm.py data/jv.train.tgs data/jv.train.txt > japanese_bigram.hmm")
    os.system("perl viterbi.pl japanese_bigram.hmm < data/jv.test.txt > japanese_bigram.out")
    os.system("python tag_acc.py data/jv.test.tgs  japanese_bigram.out")


def analyze_japanese_tri_gram():
    os.system("python trigram_hmm.py data/jv.train.tgs data/jv.train.txt data/jv.test.txt > japanese_trigram.out")
    os.system("python tag_acc.py data/jv.test.tgs  japanese_trigram.out")


if __name__ == "__main__":
    print("Running Japanese bigram")
    analyze_japanese_bi_gram()
    print("Running Japanese trigram")
    analyze_japanese_tri_gram()
