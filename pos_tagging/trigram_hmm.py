#!/usr/bin/python

"""
Implement a trigrm HMM and viterbi here. 
You model should output the final tags similar to `viterbi.pl`.

Usage:  python train_trigram_hmm.py tags text > tags

"""
import math
import sys,re
from collections import defaultdict
import random



import sys, re

from collections import defaultdict

TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]


OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"


def train_trigram_hmm(tag_file_name, token_file_name):
    tags = set()
    vocabulary = {}
    transitions_bi_gram = {}
    transitions_bi_gram_total = defaultdict(int)
    emissions = {}
    emissions_total_counts = defaultdict(int)
    trigram_transitions = {}
    trigram_tag_total = defaultdict(int)
    with open(tag_file_name) as tag_file, open(token_file_name) as token_file:
        for tagString, tokenString in zip(tag_file, token_file):
            input_tags = re.split("\s+", tagString.rstrip())
            input_tokens = re.split("\s+", tokenString.rstrip())
            pairs = zip(input_tags, input_tokens)
            t_minus_one_tag = INIT_STATE
            t_minus_two_tag = INIT_STATE
            for (tag, token) in pairs:
                if token not in vocabulary:
                    vocabulary[token] = 1
                    token = OOV_WORD
                if tag not in emissions:
                    emissions[tag] = defaultdict(int)
                if t_minus_one_tag not in transitions_bi_gram:
                    transitions_bi_gram[t_minus_one_tag] = defaultdict(int)
                if (t_minus_two_tag, t_minus_one_tag) not in trigram_transitions:
                    trigram_transitions[(t_minus_two_tag, t_minus_one_tag)] = defaultdict(int)

                emissions[tag][token] += 1
                emissions_total_counts[tag] += 1
                transitions_bi_gram[t_minus_one_tag][tag] += 1
                transitions_bi_gram_total[t_minus_one_tag] += 1
                trigram_transitions[(t_minus_two_tag, t_minus_one_tag)][tag] += 1
                trigram_tag_total[(t_minus_two_tag, t_minus_one_tag)] += 1
                t_minus_two_tag = t_minus_one_tag
                t_minus_one_tag = tag
            if t_minus_one_tag not in transitions_bi_gram:
                transitions_bi_gram[t_minus_one_tag] = defaultdict(int)
            if (t_minus_two_tag, t_minus_one_tag) not in trigram_transitions:
                trigram_transitions[(t_minus_two_tag, t_minus_one_tag)] = defaultdict(int)

            transitions_bi_gram[t_minus_one_tag][FINAL_STATE] += 1
            transitions_bi_gram_total[t_minus_one_tag] += 1

            trigram_transitions[(t_minus_two_tag, t_minus_one_tag)][FINAL_STATE] += 1
            trigram_tag_total[(t_minus_two_tag, t_minus_one_tag)] += 1

    bi_gram_transition_probability_matrix = {}
    tri_gram_transition_probability_matrix = {}
    emission_probability_matrix = {}
    unigram_count_matrix = defaultdict(lambda: 0)
    for t_minus_one_tag in transitions_bi_gram:
        for tag in transitions_bi_gram[t_minus_one_tag]:
            unigram_count_matrix[t_minus_one_tag] += 1
            unigram_count_matrix[tag] += 1
            tags.update([t_minus_one_tag])
            tags.update([tag])
            probability = float(transitions_bi_gram[t_minus_one_tag][tag]) / transitions_bi_gram_total[t_minus_one_tag]
            bi_gram_transition_probability_matrix[(t_minus_one_tag, tag)] = math.log(probability)

    for (t_minus_two_tag, t_minus_one_tag) in trigram_transitions:
        for tag in trigram_transitions[(t_minus_two_tag, t_minus_one_tag)]:
            unigram_count_matrix[t_minus_two_tag] += 1
            unigram_count_matrix[t_minus_one_tag] += 1
            unigram_count_matrix[tag] += 1
            tags.update([t_minus_two_tag])
            tags.update([t_minus_one_tag])
            tags.update([tag])
            probability = float(trigram_transitions[(t_minus_two_tag, t_minus_one_tag)][tag]) / trigram_tag_total[(t_minus_two_tag, t_minus_one_tag)]
            tri_gram_transition_probability_matrix[(t_minus_two_tag, t_minus_one_tag, tag)] = math.log(probability)


    for tag in emissions:
        tags.update([tag])
        for token in emissions[tag]:
            probability = float(emissions[tag][token]) / emissions_total_counts[tag]
            emission_probability_matrix[(tag, token)] = math.log(probability)

    return tri_gram_transition_probability_matrix, bi_gram_transition_probability_matrix, emission_probability_matrix, unigram_count_matrix, tags, vocabulary


def viterbi_trigram(evaluation_tokens, trigrams, bi_grams, emissions, uni_gram_counts, tags, vocab):
    all_states = list(tags)
    uni_grams = {}
    vocab_size = sum(uni_gram_counts.values())
    denominator_smoothing =  len(uni_gram_counts)

    vocabulary = {}
    for key in emissions:
        vocabulary[key[1]] = 1

    for tag in all_states:
        uni_grams[tag] = math.log(uni_gram_counts[tag] + 1) - math.log(vocab_size + denominator_smoothing)
    with open(evaluation_tokens) as TEXT:
        for line in TEXT:
            tokens = re.split("\s+", line.rstrip())
            sequence_length = len(tokens)
            for index, word in enumerate(tokens):
                if word not in vocabulary:
                    tokens[index] = OOV_WORD

            viterbi_records = [{}]

            for st in states:
                prob = trigrams.get((INIT_STATE, INIT_STATE, st), 0.0)
                viterbi_records[0][(INIT_STATE, st)] = {"prob": prob, "prev": INIT_STATE}

            for index in range(0, sequence_length):
                word = tokens[index]
                viterbi_records.append({})
                for current_state in all_states:
                    for t_minus_one in all_states:
                        for t_minus_two in all_states:
                            if (current_state, word) in emissions and \
                                    (t_minus_two, t_minus_one) in viterbi_records[index]:
                                if (t_minus_two, t_minus_one, current_state) in trigrams:
                                    transition_score = trigrams[(t_minus_two, t_minus_one, current_state)]
                                elif (t_minus_one, current_state) in bi_grams:
                                    transition_score = bi_grams[(t_minus_one, current_state)]
                                else:
                                    transition_score = uni_grams[current_state]
                                probability = viterbi_records[index][(t_minus_two, t_minus_one)]["prob"] + emissions[
                                    (current_state, word)] + transition_score
                                if (t_minus_one, current_state) not in viterbi_records[index + 1] or \
                                        viterbi_records[index + 1][(t_minus_one, current_state)]["prob"] < probability:
                                    viterbi_records[index + 1][(t_minus_one, current_state)] = {"prob": probability,
                                                                                                "prev": t_minus_two}

            # print(viterbi_records[0])
            goal_reached = False
            best_score = -math.inf
            current_state_max = INIT_STATE
            previous_state_max = INIT_STATE

            for current in all_states:
                for previous_state in all_states:
                    if (previous_state, current) in viterbi_records[sequence_length]:
                        if (previous_state, current, FINAL_STATE) in trigrams:
                            transition_score = trigrams[(previous_state, current, FINAL_STATE)]
                        elif (previous_state, current) in bi_grams:
                            transition_score = bi_grams[(previous_state, current)]
                        else:
                            transition_score = uni_grams[current]
                        probability = viterbi_records[sequence_length][(previous_state, current)][
                                          "prob"] + transition_score
                        if goal_reached is False or probability > best_score:
                            goal_reached = True
                            best_score = probability
                            current_state_max = current
                            previous_state_max = previous_state

            ans_list = [current_state_max, previous_state_max]
            if goal_reached:
                for i in range(sequence_length, 2, -1):
                    tag = viterbi_records[i][(previous_state_max, current_state_max)]["prev"]
                    ans_list.append(tag)
                    current_state_max = previous_state_max
                    previous_state_max = tag
                print(' '.join(reversed(ans_list)))
            else:
                print(' ')



if __name__ == "__main__":
    TRAINING_TAGS_FILE = sys.argv[1]
    TRAINING_TOKENS_FILE = sys.argv[2]
    EVALUATION_TOKENS_FILE = sys.argv[3]

    trigram_prob, bi_gram_prob, emission_prob, uni_grams, states, words = train_trigram_hmm(
        tag_file_name=TRAINING_TAGS_FILE, token_file_name=TRAINING_TOKENS_FILE)

    viterbi_trigram(EVALUATION_TOKENS_FILE, trigram_prob, bi_gram_prob, emission_prob, uni_grams, states, words)
