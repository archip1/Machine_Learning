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
Subsititution cipher frequency analysis
"""
ETAOIN = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
from sys import flags
from collections import Counter # Helpful class, see documentation or help(Counter)

def freqDict(ciphertext: str) -> dict:
    """
    Analyze the frequency of the letters
    """

    # dictionaries to store the mapping
    freqDict = {}
    mapping = {}

    # iterate thru all the letters in the message if its alphabetical, 
    # then count how many of each letter is in the message and store to freqDict
    for letter in ciphertext:
        if letter.isalpha():
            letter = letter.upper()
            if letter in freqDict:
                freqDict[letter] += 1
            else:
                freqDict[letter] = 1

    # Sort dict with most occurences at top
    # sorted_freq = sorted(freqDict.items())
    # sorted_freq = sorted(freqDict.items(), key = lambda x:x[1], reverse=True)
    sorted_freq = {k: v for k, v in sorted(freqDict.items(), key=lambda item: (-item[1], item[0]))}

    # find the letter with the most occurences to assign it the letter that occurs rthe most in the english dictionary according to ETAOIN
    # iterate thru the freqDict and keep popping 
    count = 0
    for i in sorted_freq:
        char = i[0]
        mapping[char] = ETAOIN[count]
        count += 1

    return mapping

def freqDecrypt(mapping: dict, ciphertext: str) -> str:
    """
    Apply the mapping to ciphertext
    """
    plain_text = ""
    # iterate thru the text and map each letter according to the dictionary based on frequency
    for letter in ciphertext:
        if letter in mapping:
            plain_text += mapping[letter]
        else:
            plain_text += letter
    return plain_text

def test():
    "Run tests"
    assert type(freqDict("A")) is dict
    assert freqDict("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")["A"] == "E"
    assert freqDict("AABBA")['B'] == "T"
    assert freqDict("-: AB CD AH")['A'] == "E"

    assert(freqDict("MKLAKAALK") == {'A': 'E', 'K': 'T', 'L': 'A', 'M': 'O'})
    assert(freqDict("D64X0CIK0LW2AC8Y2APEKK4OR550HC?YF ") == {'C': 'E', 'K': 'T', 'A': 'A', 'Y': 'O', 'D': 'I', 'E': 'N', 'F': 'S', 'H': 'H', 'I': 'R', 'L': 'D', 'O': 'L', 'P': 'C', 'R': 'U', 'W': 'M', 'X': 'W'})
    assert(freqDict("X8IJZ77LRFUL1 1.BMAN743IKHLFI.NTM9 1E9KNJ6O 2.JP7Y") == {'I': 'E', 'J': 'T', 'L': 'A', 'N': 'O', 'F': 'I', 'K': 'N', 'M': 'S', 'A': 'H', 'B': 'R', 'E': 'D', 'H': 'L', 'O': 'C', 'P': 'U', 'R': 'M', 'T': 'W', 'U': 'F', 'X': 'G', 'Y': 'Y', 'Z': 'P'})
    assert(freqDict("D6AFYC?7I?EJUFS11DWKY8!8AMGFU0AWMX72PCZGE.ZTR.V6FH276NU4.BK7.W5J R0J?N0HX7J6CQFN?JVX39NS5") == {'F': 'E', 'J': 'T', 'N': 'A', 'A': 'O', 'C': 'I', 'U': 'N', 'W': 'S', 'X': 'H', 'D': 'R', 'E': 'D', 'G': 'L', 'H': 'C', 'K': 'U', 'M': 'M', 'R': 'W', 'S': 'F', 'V': 'G', 'Y': 'Y', 'Z': 'P', 'B': 'B', 'I': 'V', 'P': 'K', 'Q': 'J', 'T': 'X'})
    assert(freqDict("GCRJZMQYY6ENS644KF18SYQCSIPIJ?Z4U?HC08SHX0T3?2LZ?IFBRF5.D! 8PI59J9ZICAFRW  T2M6PY 3JJ2KG0BGS3IQ0179") == {'I': 'E', 'J': 'T', 'S': 'A', 'C': 'O', 'F': 'I', 'Y': 'N', 'Z': 'S', 'G': 'H', 'P': 'R', 'Q': 'D', 'R': 'L', 'B': 'C', 'H': 'U', 'K': 'M', 'M': 'W', 'T': 'F', 'A': 'G', 'D': 'Y', 'E': 'P', 'L': 'B', 'N': 'V', 'U': 'K', 'W': 'J', 'X': 'X'})
    
    assert freqDecrypt({"A": "E", "Z": "L", "T": "H", "F": "O", "U": "W", "I": "R", "Q": "D"}, "TAZZF UFIZQ!") == "HELLO WORLD!"

# Invoke test() if called via `python3 a5p1.py`
# but not if `python3 -i a5p1.py` or `from a5p1 import *`
if __name__ == '__main__' and not flags.interactive:
    test()
