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
Problem 1
"""

from sys import flags
# LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '

def readFile(filename):
    #  Read file and append letters to a list

    file = open(filename, 'r')
    text = (file.read()).upper()
    file.close()

    textList = []
    for i in text:
        textList.append(i)

    return(textList)

def ngramsFreqsFromFile(textFile, n: int) -> dict:
    """
    textFile: 'wells.txt'
    """
    textList = readFile(textFile)
    
    ngram = []
    # create a list with letters as many times as the n gram
    loop = (len(textList)-n+1)
    for i in range(loop):
        for j in range(n):
            try:
                ngram.append(textList[j+i]) # Append the current ngram into the nGram list. 
            except:
                continue
    
    # Join the ngram characters into one single ngram
    i = 0
    for j in range(int(len(ngram)/n)):
        # Below is the link that was used for this:
        # https://www.geeksforgeeks.org/python-merge-list-elements/
        ngram[i : n+i] = [''.join(ngram[i : n+i])]
        i = i+1

    # Create a the Dictionary with frequency of n grams
    ngramDict={}
    for item in ngram:
        if item in ngramDict:
            value = ((ngramDict.get(item))*len(ngram)) # Find out the percent of time it occurs in ngramDict
            newValue = (value+1)/len(ngram)
            ngramDict[item] = newValue # Update the dict
        else:
            # if the ngram is not in the dict, put it in.
            value = (1/len(ngram))  
            ngramDict[item] = value

    return(ngramDict)

def test():
    "Run tests"
    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking

if __name__ == "__main__" and not flags.interactive:
    test()
