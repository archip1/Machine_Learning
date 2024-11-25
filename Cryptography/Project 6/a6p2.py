#!/usr/bin/env python3

#---------------------------------------------------------------
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
#---------------------------------------------------------------

"""
Problem 2
"""

from sys import flags
from collections import Counter

def keyScore( mapping: dict, ciphertext: str, frequencies: dict, n: int ) -> float:

    # Decipher the message with the mapping we have
    deciphered_text = ""
    for char in ciphertext:
        if char == ' ':
            deciphered_text += ' '
        else:
            deciphered_text += mapping[char]

    ngram_score = 0.0

    # iterate thru each letter and get its relative frequency and add it to the n_gram score variable
    for i in range(len(deciphered_text) - n + 1):
        ngram = deciphered_text[i:i + n]
        if ngram in frequencies:
            ngram_score += frequencies[ngram]

    # # another deciphering list    
    # ngramDecipher = []
    # loop = (len(deciphered_text)-n+1)

    # for i in range(loop):
    #     for j in range(n):
    #         try: # if it goes out of range, then that means it is done, so just continue
    #             ngramDecipher.append(deciphered_text[j+i]) # Append to ngramDecipher.
    #         except:
    #             continue

    # i = 0
    # for j in range(int(len(ngramDecipher)/n)):
    #     ngramDecipher[i : n+i] = [''.join(ngramDecipher[i : n+i])] # Join the characters in ngrams strings
    #     i = i+1

    # score = 0 # currect score is 0
    # for key,value in frequencies.items():
    #     c = 0 # find out how many times it occurs
    #     while key in ngramDecipher:
    #         c += 1 # if the key appears in the list
    #         ngramDecipher.remove(key) # After incrementing it, remove it from the list so it is not counted twice.
    #     score += c*value # Use the formula given to us. NewScore = currectscore + (c(g)*f(g))

    return(ngram_score)
  

def test():
    "Run tests"
    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking

if __name__ == "__main__" and not flags.interactive:
    test()





