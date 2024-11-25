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
Assignment 7 Problem 1
"""

from sys import flags
import vigenereCipher
from itertools import groupby
import math
LOOP = True

def ngrams(plaintext, n: int) -> dict:
    # from assign 6, find trigrams of the plain text and return the dictionary
    ngram = []
    num_loops = (len(plaintext)-n+1)

    for i in range(num_loops):
        for j in range(n):
            try:
                ngram.append(plaintext[j+i])
            except:
                continue

    i = 0
    for j in range(int(len(ngram)/n)):
        ngram[i : n+i] = [''.join(ngram[i : n+i])]
        i = i+1

    return ngram

def ngram_dictionary(ngram):
    # from assign 6, create a dictionary of all occurences of trigrams
    ngramDict={}
    for item in ngram:
        if item in ngramDict:
            ngramDict[item] += 1
        else:
            ngramDict[item] = 1

    ngramDict = sorted(ngramDict.items(), key=lambda x: x[1], reverse=True)

    return ngramDict

def only_keep_repeated_ngrams(ngramDict):
    # this iterates thru the dictionary and only keeps the trigrams that occur more than one in the plain text
    repeated_words = {}
    for key, value in ngramDict:
        if value != 1:
            repeated_words[key] = value

    return(repeated_words)

def get_pattern(key, plaintext):
    # gives u a list with the index of the letter of the key that will encipher each letter of the plain text
    #  for the given example THOSE encrypted by WICK, the function will return [01230]
    pattern_list = []
    number = 0
    for i in range(len(plaintext)):
        if number == len(key):
            number = 0
        pattern_list.append(number)
        number += 1
    return (pattern_list)

def get_pattern_of_repeated_words(repeated_words, plaintext, pattern):
    # this function will return a dictionary of dictionaries in which the key will be the trigram
    # the value will be a dictionary containing the index as the key where the trigram starts and the value will be the pattern of the trigram at that index
    repeated_trigrams = {}
    for i in range(len(plaintext)):
        trigram = i + 3
        if plaintext[i:trigram] in repeated_words:
            if plaintext[i:trigram] in repeated_trigrams:
                repeated_trigrams[plaintext[i:trigram]][i] = pattern[i:trigram]
            else:
                index_tracker = {}
                index_tracker[i] = pattern[i:trigram]
                repeated_trigrams[plaintext[i:trigram]] = index_tracker
                
    return(repeated_trigrams)

def find_insert_index(trigram_patterns, plaintext, last_append):
    # This function checks if the tri gram repeats, if it does, if the next occurence of that trigram has the same pattern it will return the starting index of the first repeating trigram to add an x 4 indecies after
    global LOOP
    iterate = True
    for i in range(len(plaintext)):
        if iterate:
            trigram = i + 3
            if i >= last_append:
                if plaintext[i:trigram] in trigram_patterns:
                    index = list(trigram_patterns[plaintext[i:trigram]].keys())
                    if len(index) == 2:
                        if (trigram_patterns[plaintext[i:trigram]][index[0]] == trigram_patterns[plaintext[i:trigram]][index[1]]):
                            insert_index = i
                            iterate = False
                            break
                    else:
                        compare = trigram_patterns[plaintext[i:trigram]][i]
                        compareBool = True
                        for j in range(len(index)):
                            if index[j] > i and compare == trigram_patterns[plaintext[i:trigram]][index[j]]:
                                insert_index = i
                                iterate = False
                                compareBool = False
                                break
                        if compareBool:
                            LOOP = False
                            insert_index = None
                            iterate = False
        else:
            break

    return(insert_index)

def insert_x(insert_index, plaintext, last_append):
    # insert an x 4 indecies after the index provided
    temp = list(plaintext)
    j = insert_index + 3

    if insert_index >= last_append and j<len(plaintext):
        temp.insert(j, "X")
        last_append = insert_index + 4
    
    return (''.join(temp), last_append)

def antiKasiski(key: str, plaintext: str):
    """
    Thwart Kasiski examination 
    """
    # keep track of iterations
    global LOOP
    LOOP = True

    # get ngrams of text, create a dictionary of occurences and only keep trigrams that are repeated
    ngram = ngrams(plaintext, 3)
    ngram_Dict = ngram_dictionary(ngram)
    repeated_words = only_keep_repeated_ngrams(ngram_Dict)

    last_append = 0
    while LOOP:
        # get the pattern (which letter of the key you would encrypt the plaintext with) and assign to each trigram with the index value as a dictionary od dictionaries
        pattern = get_pattern(key, plaintext)
        trigram_patterns = get_pattern_of_repeated_words(repeated_words, plaintext, pattern)

        # find where to insert the x and insert it
        insert_place = find_insert_index(trigram_patterns, plaintext, last_append)
        if insert_place != None:
            plaintext, last_append = insert_x(insert_place, plaintext, last_append)

    # encipher the plain text
    return(vigenereCipher.encryptMessage(key,(''.join(plaintext))))

def test():
    "Run tests"
    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking
    assert( antiKasiski('WICK', 'THOSEPOLICEOFFICERSOFFEREDHERARIDEHOMETHEYTELLTHEMAJOKETHOSEBARBERSLENTHERALOTOFMONEY')== 'PPQCAXQVEKGYBNZSYMTCTWHPAZGNDMTKNQFODWOOPPGHUBGVHBJOTUCTKSGDDWUOXITLAZUVAVVRAZCVKBQPIWPOU')
    assert (antiKasiski("WICK", "ABCDABCD") == 'WJEHZIDMZ')
    # print (antiKasiski("HJK", "AAABBBBBBBBBAAA") == 'HJKEGLIKHIKLEKLIJKH')


# Invoke test() if called via `python3 a7p1.py`
# but not if `python3 -i a7p1.py` or `from a7p1 import *`
if __name__ == '__main__' and not flags.interactive:
    test()
