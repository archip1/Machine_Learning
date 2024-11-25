#!/usr/bin/env python3

#---------------------------------------------------------------
#
# CMPUT 331 Student Submission License
# Version 1.0
# Copyright 2023 <<Insert your name here>>
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
Problem 4
"""

from sys import flags
import a6p3
import a6p1
from collections import Counter # Helpful class, see documentation or help(Counter)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ETAOIN = "ETAOINSHRDLCUMWFGYPBVKJXQZ"

# First 2 functions from assign 5!
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

    ciphertextList = [] # Conver the string to list and uppercase all the characters
    for elements in ciphertext:
        ciphertextList.append(elements.upper())


    decipheredList = []     # initialize a list
    for i in ciphertextList:
        if i in LETTERS: # If the currect character is a Letter
            if i in ciphertextList: # And i is in the character's list
                index =  ciphertextList.index(i) # Find out it's index in ciphertextList
                decipheredList.append(mapping.get(i)) # Append the value based on the mapping
                ciphertextList.remove(i)              # remove the i from the ciphered list
                ciphertextList.insert(index, "_")     # and insert a black there instead so that
                                                      # in next loop it is not recognize.
        else:           # if I is anything other than a letter
            decipheredList.append(i) # then just append it without minipulating it
    deciphered = "".join(decipheredList) # join the deciphered list

    return deciphered # And return it

def readFile(filename):
    #  Read file and append letters to a list

    file = open(filename, 'r')
    text = (file.read()).upper()
    file.close()

    textList = []
    for i in text:
        textList.append(i)

    return(textList)

def breakSub(cipherFile, textFile, n: int) -> None:
    """
    Inputs:
        cipherFile: 
            'text_finnegan_cipher.txt' for implementation
            'text_cipher.txt' for submission
        textFile: 'wells.txt'
    Outputs:
        'text_finnegan_plain.txt' for implementation
        'text_plain.txt' for submission
    """

    # read file
    ciphertext = readFile(cipherFile)

    # Mapping using frequency analysis
    mapping = freqDict(ciphertext)
    # add the space character mapping to itself
    mapping[" "] = " "

    # get n gram frequency
    frequencies = a6p1.ngramsFreqsFromFile(textFile, n)
    
    # keep running best successor 
    while True:
        BestMapping = a6p3.bestSuccessor(mapping, ciphertext, frequencies, n) # Obtains the best mapping
        if BestMapping == mapping:
            break
        mapping = BestMapping
    accepted_mapping = mapping

    # Decipher the ciphered text using the bestMapping
    BestDecipherment = freqDecrypt(accepted_mapping, ciphertext)

    #Best deciphered text possible using the Hill-Climbing Substitution cipher

    # Wite to file
    file = open("text_plain.txt", 'w')
    file.write(BestDecipherment)
    file.close()

    return None
        
def test():
    "Run tests"
    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking
    breakSub("text_cipher.txt", "wells.txt", 3)


if __name__ == "__main__" and not flags.interactive:
    test()