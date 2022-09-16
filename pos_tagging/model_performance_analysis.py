import sys
import sys, re
import math
from collections import defaultdict
import subprocess
import matplotlib.pyplot as plt
import os

def load_training_data(TRAINING_TAGS_FILE, TRAINING_TOKENS_FILE, no_of_lines):
    vocab = {}
    OOV_WORD = "OOV"
    INIT_STATE = "init"
    FINAL_STATE = "final"

    emissions = {}
    transitions = {}
    transitionsTotal = defaultdict(int)
    emissionsTotal = defaultdict(int)
    current_line = 0
    with open(TRAINING_TAGS_FILE) as tagFile, open(TRAINING_TOKENS_FILE) as tokenFile:
        tag_file = tagFile.readlines()[:no_of_lines]
        token_file = tokenFile.readlines()[:no_of_lines]
        print("Length of tag file :{}, Length of token file:{}".format(len(tag_file), len(token_file)))
        for tagString, tokenString in zip(tag_file, token_file):
            tags = re.split("\s+", tagString.rstrip())
            tokens = re.split("\s+", tokenString.rstrip())
            pairs = list(zip(tags, tokens))

            prevtag = INIT_STATE

            for (tag, token) in pairs:
                if token not in vocab:
                    vocab[token] = 1
                    token = OOV_WORD

                if tag not in emissions:
                    emissions[tag] = defaultdict(int)
                if prevtag not in transitions:
                    transitions[prevtag] = defaultdict(int)
                emissions[tag][token] += 1
                emissionsTotal[tag] += 1
                transitions[prevtag][tag] += 1
                transitionsTotal[prevtag] += 1
                prevtag = tag

            if prevtag not in transitions:
                transitions[prevtag] = defaultdict(int)

            transitions[prevtag][FINAL_STATE] += 1
            transitionsTotal[prevtag] += 1

    # transition_prob = {}
    # emission_prob = {}

    output_file = "bigram.hmm"
    results = open(output_file, "w")
    for prevtag in transitions:
        for tag in transitions[prevtag]:
            results.write("trans {} {} {} ".format(prevtag, tag, float(transitions[prevtag][tag]) / transitionsTotal[prevtag]) + "\n")
            # transition_prob[(prevtag, tag)] = math.log(probability)

    for tag in emissions:
        for token in emissions[tag]:
            probability = float(emissions[tag][token]) / emissionsTotal[tag]
            results.write("emit {} {} {} ".format(tag, token, float(emissions[tag][token]) / emissionsTotal[tag]) + "\n")
            # emission_prob[(tag, token)] = math.log(probability)

    # return transition_prob, emission_prob


def run_bi_gram_viterbi():
    os.system('perl viterbi.pl bigram.hmm < data/ptb.22.txt > predictions_ptb22.out')
    # subprocess.run(["perl", "viterbi.pl", "bigram.hmm", "data/ptb.22.txt", "predictions_ptb22.out"])


def __evaluate_tag_acc(golds, hypos):
        tag_errors = 0
        sent_errors = 0
        tag_tot = 0
        sent_tot = 0

        for g, h in zip(golds, hypos):
            g = g.strip()
            h = h.strip()

            g_toks = re.split("\s+", g)
            h_toks = re.split("\s+", h)

            error_flag = False

            for i in range(len(g_toks)):
                if i >= len(h_toks) or g_toks[i] != h_toks[i]:
                    tag_errors += 1
                    error_flag = True

                tag_tot += 1

            if error_flag:
                sent_errors += 1

            sent_tot += 1
        error_rate_word = float(tag_errors) / tag_tot
        error_rate_sentence = float(sent_errors) / sent_tot
        return error_rate_word, error_rate_sentence


def get_error_rate_metrics():
    GOLD_FILE = "data/ptb.22.tgs"
    HYPO_FILE = "predictions_ptb22.out"
    with open(GOLD_FILE) as goldFile, open(HYPO_FILE) as hypoFile:
        golds = goldFile.readlines()
        hypos = hypoFile.readlines()

        if len(golds) != len(hypos):
            raise ValueError("Length is different for two files!")

    error_rate_word, error_rate_sentence = __evaluate_tag_acc(golds, hypos)
    return error_rate_word, error_rate_sentence


def draw_graph(training_set_size, word_error_rate, sentence_error_rate):
    word_error_rate_fig = plt.figure()
    ax1 = word_error_rate_fig.add_subplot(111)
    ax1.plot(training_set_size, word_error_rate)
    ax1.set_xlabel("Size of Training Data")
    ax1.set_ylabel("Word Error Rate")
    word_error_rate_fig.savefig("Word Error Rate")

    sentence_error_rate_fig = plt.figure()
    ax2 = sentence_error_rate_fig.add_subplot(111)
    ax2.plot(training_set_size, sentence_error_rate)
    ax2.set_xlabel("Size of Training Data")
    ax2.set_ylabel("Sentence Error Rate")
    sentence_error_rate_fig.savefig("Sentence Error Rate")


if __name__ == "__main__":
    TRAINING_TAGS_FILE = "data/ptb.2-21.tgs"
    TRAINING_TOKENS_FILE = "data/ptb.2-21.txt"
    EVALUATION_TOKENS_FILE = "data/ptb.22.txt"
    GROUND_TRUTH_FILE = "data/ptb.22.tgs"
    no_of_lines_list = [i * 5000 for i in range(1, 9, 1)]

    word_error_rate = []
    sentence_error_rate = []
    for n in no_of_lines_list:
        print("N is {}".format(n))
        load_training_data(TRAINING_TAGS_FILE, TRAINING_TOKENS_FILE, n)
        run_bi_gram_viterbi()
        word, sentence = get_error_rate_metrics()
        word_error_rate.append(word)
        sentence_error_rate.append(sentence)
    draw_graph(training_set_size=no_of_lines_list, word_error_rate=word_error_rate,
               sentence_error_rate=sentence_error_rate)

