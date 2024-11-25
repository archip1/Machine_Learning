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
Problem 2
"""

from sys import flags
import numpy as np

def deciphermentAccuraccy(text1: str, text2: str):
    correct = 0
    total = 0

    # iterate thru the text and count how many char are correct
    for i in range(len(text1)):
        if text1[i].isalpha():
            total += 1
            if text1[i].lower() == text2[i].lower():
                correct += 1
    
    # dived correct characters by total num of charachters
    return (correct / total)

def keyAccuracy(text1: str, text2: str):
    charCount = {}
    # track repetition
    # iterate thru text and count how many letters 
    for i in range(len(text1)):
        if text1[i].isalpha():
            if text1[i].lower() in charCount:
                charCount[text1[i].lower()] += 1
            else:
                charCount[text1[i].lower()] = 1
    
    # find all wrong answers and iterate thru the text and whatever is incorrect, count it
    # dont count it as wrong if it is already counted
    wrong = []
    for i in range(len(text1)):
        if text1[i].isalpha():
            if text1[i].lower() != text2[i].lower():
                if text1[i].lower() not in wrong:
                    wrong.append(text1[i].lower())

    # return the correct letters out of totals
    incorrect = len(wrong)
    total = len (charCount)
    correct = total - incorrect

    return (correct / total)

def evalDecipherment(text1: str, text2: str) -> [float, float]:
    """
    docstring
    """
    # call the key accuracy and decipherment accuracy functions
    key = keyAccuracy(text1, text2)
    decipherment = deciphermentAccuraccy(text1, text2)

    return(key, decipherment)

def test():
    "Run tests"
    # print(evalDecipherment("this is an example", "tsih ih an ezample"))
    np.testing.assert_array_almost_equal(evalDecipherment("this is an example", "tsih ih an ezample") , [0.7272727272727273, 0.7333333333333333])
    np.testing.assert_almost_equal(evalDecipherment("the most beautiful course is 331!", "tpq munt bqautiful cuurnq in 331!") , [0.7142857142857143, 0.625])
if __name__ == '__main__' and not flags.interactive:
    test()
