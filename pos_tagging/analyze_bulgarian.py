import os


def analyze_bulgarian_bi_gram():
    os.system("python train_hmm.py data/btb.train.tgs data/btb.train.txt > bulgarian_bigram.hmm")
    os.system("perl viterbi.pl bulgarian_bigram.hmm < data/btb.test.txt > bulgarian_bigram.out")
    os.system("python tag_acc.py data/btb.test.tgs  bulgarian_bigram.out")


def analyze_bulgarian_tri_gram():
    os.system("python trigram_hmm.py data/btb.train.tgs data/btb.train.txt data/btb.test.txt > bulgarian_trigram.out")
    os.system("python tag_acc.py data/btb.test.tgs  bulgarian_trigram.out")


if __name__ == "__main__":
    print("Running Bulgarian bigram")
    analyze_bulgarian_bi_gram()
    print("Running Bulgarian trigram")
    analyze_bulgarian_tri_gram()
