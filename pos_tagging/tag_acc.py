#!/usr/bin/python

"""
Calculates and prints out error rate (word-level and sentence-level) of a POS tagger.

Usage: python tag_acc.py gold-tags hypothesized-tags

Tags should be separated by whitespace, no leading or trailing spaces,
one sentence per line.  There's no error handling if things don't line up!
"""


import re
import sys


def evalaute_tag_acc(golds, hypos):
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

    print("error rate by word:      ", tag_errors / tag_tot, f" ({tag_errors} errors out of {tag_tot})")
    print("error rate by sentence:  ", sent_errors / sent_tot, f" ({sent_errors} errors out of {sent_tot})")


if __name__ == "__main__":
    # arguemnt
    GOLD_FILE = sys.argv[1]
    HYPO_FILE = sys.argv[2]

    with open(GOLD_FILE) as goldFile, open(HYPO_FILE) as hypoFile:
        golds = goldFile.readlines()
        hypos = hypoFile.readlines()

        if len(golds) != len(hypos):
            raise ValueError("Length is different for two files!")

    evalaute_tag_acc(golds, hypos)
