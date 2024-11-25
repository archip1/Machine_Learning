#!/usr/bin/python3

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
CMPUT 331 Assignment 2 Student Solution
September 2023
Author: <Archi Patel>
"""

from typing import List

"""
function: encryptMessage

Encrypts plain text with the columnar transposition cipher using the key to append that row in that order 

Arguements: 
    - key: a list of integers which determine the order in which the columns are to be rearranged
    - message: a plaintext string to be encrypted

Returns: 
    - the message encrypted with columnar transposition cipher with the rows joined using the key

"""
def encryptMessage(key: List[int], message: str):
    # From the key we can assume there are that many columns
    numcolumns = len(key)

    # Each string in ciphertext represents a column in the grid.
    ciphertext = [''] * numcolumns

    # Loop through each column in ciphertext.
    for column in range(numcolumns):
        currentIndex = column

        # Keep looping until currentIndex goes past the message length.
        while currentIndex < len(message):
            # Place the character at currentIndex in message at the
            # end of the current column in the ciphertext list.
            ciphertext[column] += message[currentIndex]

            # move currentIndex over
            currentIndex += numcolumns

    # order the cipher text in terms of the key
    orderedciphertext = ""
    for i in key:
        orderedciphertext += ciphertext[i-1]

    # Convert the ciphertext list into a single string value and return it.
    return orderedciphertext

def test():
    assert encryptMessage([2, 4, 1, 5, 3], "CIPHERS ARE FUN") == "IS HAUCREERNP F"
    assert encryptMessage([1, 3, 2], "ABCDEFG") == "ADGCFBE"
    assert encryptMessage([2, 1], "HELLO WORLD") == "EL OLHLOWRD"

    # a = encryptMessage([2, 4, 1, 5, 3], "CIPHERS ARE FUN")
    # b = encryptMessage([1, 3, 2], "ABCDEFG")
    # c = encryptMessage([2, 1], "HELLO WORLD")

    # print(a)
    # print(b)
    # print(c)

from sys import flags

if __name__ == "__main__" and not flags.interactive:
    test()
