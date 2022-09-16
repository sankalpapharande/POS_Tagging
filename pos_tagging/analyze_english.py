import os


def analyze_english_bi_gram():
    os.system("python train_hmm.py data/ptb.2-21.tgs data/ptb.2-21.txt > english_bigram.hmm")
    os.system("perl viterbi.pl english_bigram.hmm < data/ptb.22.txt > english_bigram.out")
    os.system("python tag_acc.py data/ptb.22.tgs  english_bigram.out")


def analyze_english_tri_gram():
    os.system("python trigram_hmm.py data/ptb.2-21.tgs data/ptb.2-21.txt data/ptb.22.txt > english_trigram.out")
    os.system("python tag_acc.py data/ptb.22.tgs  english_trigram.out")


if __name__ == "__main__":
    print("Running English bigram")
    analyze_english_bi_gram()
    print("Running English trigram")
    analyze_english_tri_gram()
