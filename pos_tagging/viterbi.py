#!/usr/bin/python

"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""

import math
import re
import sys


OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"


def preprocess(hidden_markov_models):
    vocab = {}
    transition_prob = {}
    emission_prob = {}
    with open(hidden_markov_models) as hmm_file:
        states = []
        count_trans = 0
        count_emit = 0
        for line in hmm_file:
            _type, tag1, tag2, p = line.split()
            key = (tag1, tag2)
            if _type == 'trans':
                count_trans += 1
                states.append(tag1)
                states.append(tag2)
                transition_prob[key] = math.log(float(p))
            elif _type == "emit":
                count_emit += 1
                states.append(tag1)
                vocab[tag2] = 1
                emission_prob[key] = math.log(float(p))

    states.append(FINAL_STATE)
    states = list(set(states))
    vocab = list(set(vocab))
    return emission_prob, transition_prob, states, vocab


def viterbi_algorithm(token_file, emission_prob, transition_prob, states, vocab):
    # with open(token_file) as TEXT:
    for line in token_file:
        tokens = re.split("\s+", line.rstrip())
        sequence_length = len(tokens)
        for index, word in enumerate(tokens):
            if word not in vocab:
                tokens[index] = OOV_WORD
        viterbi_records = [{}]
        for st in states:
            prob = transition_prob.get((INIT_STATE, st), 0.0) + emission_prob.get((st, tokens[0]), 0.0)
            viterbi_records[0][st] = {"prob": prob, "prev": "init"}
        for index in range(0, sequence_length):
            word = tokens[index]
            viterbi_records.append({})
            for current_state in states:
                for previous_state in states:
                    if (current_state, word) in emission_prob and (
                            previous_state, current_state) in transition_prob and \
                            previous_state in viterbi_records[index]:
                        current = viterbi_records[index][previous_state]["prob"] + \
                                  transition_prob[(previous_state, current_state)] + \
                                  emission_prob[(current_state, word)]
                        if current_state not in viterbi_records[index + 1] or current > \
                                viterbi_records[index + 1][current_state]["prob"]:
                            viterbi_records[index + 1][current_state] = {"prob": current, "prev": previous_state}
        goal_reached = False
        best_score = -math.inf
        tag = INIT_STATE
        for each_tag in states:
            if (each_tag, FINAL_STATE) in transition_prob and each_tag in viterbi_records[sequence_length]:
                probability = viterbi_records[sequence_length][each_tag]["prob"] + transition_prob[
                    (each_tag, FINAL_STATE)]
                if goal_reached is False or probability > best_score:
                    goal_reached = True
                    best_score = probability
                    tag = each_tag
        ans_list = [tag]
        if goal_reached:
            for i in range(sequence_length, 1, -1):
                tag = viterbi_records[i][tag]["prev"]
                ans_list.append(tag)
            print(' '.join(reversed(ans_list)))
        else:
            print(' ')


if __name__ == "__main__":
    HMM_FILE = sys.argv[1]
    tokens = sys.stdin.readlines()
    emissions, transitions, all_states, vocabulary = preprocess(HMM_FILE)
    viterbi_algorithm(tokens, emissions, transitions, all_states, vocabulary)
