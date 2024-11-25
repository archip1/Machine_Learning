#!/usr/bin/env python3

# ---------------------------------------------------------------
#
# CMPUT 331 Student Submission License
# Version 1.0
# Copyright 2023 <<Archi Patel>>
#
# Redistribution is forbidden in all circumstances. Use of this software
# without explicit authorization from the author is prohibited.
#
# This software was produced as a solution for an assignment in the course
# CMPUT 331 - Computational Cryptography at the University of
# Alberta, Canada. This solution is confidential and remains confidential
# after it is submitted for grading.
#
# Copying any part of this solution without including this copyright notice
# is illegal.
#
# If any portion of this software is included in a solution submitted for
# grading at an educational institution, the submitter will be subject to
# the sanctions for plagiarism at that institution.
#
# If this software is found in any public website or public repository, the
# person finding it is kindly requested to immediately report, including
# the URL or other repository locating information, to the following email
# address:
#
#          gkondrak <at> ualberta.ca
#
# ---------------------------------------------------------------

"""
Assignment 7 Problems 2, 3, and 4
"""

import re
from sys import flags


def stringIC(text: str):
    """
    Compute the index of coincidence (IC) for text
    """
    # create a dictionary of all individual letters and how often they occur
    letters = {}
    for i in text:
        if i in letters:
            letters[i] += 1
        else:
            letters[i] = 1

    # for the numerator iterte through all the values and numerator = c*(c-1)
    numerator = 0
    for i in letters:
        numerator += letters[i] * (letters[i]- 1)
    
    # denominator is total charachters *(total char - 1)
    denominator = len(text) * (len(text)-1)

    # calculate ic
    ic = numerator/denominator

    return ic

def subseqIC(ciphertext: str, keylen: int):
    """
    Return the average IC of ciphertext for 
    subsequences induced by a given a key length
    """

    # iterate thru the keylen, get a subsequence and calculate its ic using previuous function. Keep adding to the sum
    sumic = 0
    for i in range(keylen):
        subsequence = getNthSubkeysLetters(i+1, keylen, ciphertext)
        sumic += stringIC(subsequence)
            
    # return the avgic
    return sumic/keylen

# Sort the dictionary by values from high to low and keys from low to high from geeks for geeks or chatgpt (how to sort)
def custom_sort(item):
    key, value = item
    return (-value, key)

def keyLengthIC(ciphertext: str, n: int):
    """
    Return the top n keylengths ordered by likelihood of correctness
    Assumes keylength <= 20
    """
    # calculate the ic for each key length and store it in a dictionary
    scores = {}
    for i in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20):
        score = subseqIC(ciphertext, i)
        scores[i] = score

    # Sort the dictionary by values from high to low and keys from low to high from geeks for geeks or chatgpt (how to sort)
    sorted_dict = dict(sorted(scores.items(), key=custom_sort))

    # return the top n highest amounts
    top_n = []
    count = 0
    for key in sorted_dict.keys():
        if count < n:
            top_n.append(key)
            count += 1
        else:
            break

    return top_n


def getNthSubkeysLetters(nth: int, keyLength: int, message: str):
    # Returns every nth letter for each keyLength set of letters in text.
    # E.g. getNthSubkeysLetters(1, 3, 'ABCABCABC') returns 'AAA'
    #      getNthSubkeysLetters(2, 3, 'ABCABCABC') returns 'BBB'
    #      getNthSubkeysLetters(3, 3, 'ABCABCABC') returns 'CCC'
    #      getNthSubkeysLetters(1, 5, 'ABCDEFGHI') returns 'AF'

    # Use a regular expression to remove non-letters from the message:
    message = re.compile('[^A-Z]').sub('', message)

    i = nth - 1
    letters = []
    while i < len(message):
        letters.append(message[i])
        i += keyLength
    return ''.join(letters)

def test():
    "Run tests"
    assert stringIC("ABA") == 1 / 3
    assert subseqIC('PPQCAXQVEKGYBNKMAZUHKNHONMFRAZCBELGRKUGDDMA', 3) == 0.03882783882783883
    assert subseqIC('PPQCAXQVEKGYBNKMAZUHKNHONMFRAZCBELGRKUGDDMA', 4) == 0.0601010101010101
    assert(subseqIC('PPQCAXQVEKGYBNKMAZUHKNHONMFRAZCBELGRKUGDDMA', 5) == 0.012698412698412698)
    assert(keyLengthIC('PPQCAXQVEKGYBNKMAZUYBNGBALJONITSZMJYIMVRAGVOHTVRAUCTKSGDDWUOXITLAZUVAVVRAZCVKBQPIWPOU', 5) == [8, 16, 4, 12, 6])
    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking


# Invoke test() if called via `python3 a7p234.py`
# but not if `python3 -i a7p234.py` or `from a7p234 import *`
if __name__ == '__main__' and not flags.interactive:
    test()
