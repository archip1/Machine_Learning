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
function: decryptMessage

Decrypts cipher text with the columnar transposition cipher using the key as the number of rows
The transposition decrypt function will simulate the "columns" and
"rows" of the grid that the plaintext is written on by using a list
of strings. First, we need to calculate a few values.

Arguements: 
    - key: the list of integers that was used to encrypt the original plaintext
    - message: a cipher string to be decrypted

Returns: 
    - the message decrypted with columnar transposition cipher that reverses the encrypt function in p2

"""
def decryptMessage(key: List[int], message: str):

    # The number of "columns" in our transposition grid:
    numcolumns = len(key)

    # The number of "rows" in our grid will need:
    numOfRows = int(math.ceil(len(message) / float(numcolumns)))

    # The number of "shaded boxes" in the last "column" of the grid:
    numOfShadedBoxes = (numcolumns * numOfRows) - len(message)

    # Each string in transposedtext represents a column in the grid.
    transposedtext = [''] * numcolumns

    # Seperate all the combined columns
    numletter = 0
    numfullcols = numcolumns - numOfShadedBoxes

    for column in key: 
        if column <= numfullcols:
            for j in range(numOfRows):
                transposedtext[column-1] += message[numletter]
                numletter += 1
        else:
            for j in range(numOfRows-1):
                transposedtext[column-1] += message[numletter]
                numletter += 1


    transposedmsg = ''.join(transposedtext)

    # The column and row variables point to where in the grid the next
    # character in the encrypted message will go.
    column = 0
    row = 0

    plaintext = [''] * numcolumns

    # The transposition decrypt function will simulate the "columns" and
    # "rows" of the grid that the plaintext is written on by using a list
    # of strings. First, we need to calculate a few values.

    # The number of "columns" in our transposition grid:
    numOfColumns = int(math.ceil(len(message) / float(numcolumns)))
    # The number of "rows" in our grid will need:
    numOfRows = numcolumns
    # The number of "shaded boxes" in the last "column" of the grid:
    numOfShadedBoxes = (numOfColumns * numOfRows) - len(message)

    # Each string in plaintext represents a column in the grid.
    plaintext = [''] * numOfColumns

    # The column and row variables point to where in the grid the next
    # character in the encrypted message will go.
    column = 0
    row = 0

    for symbol in transposedmsg:
        plaintext[column] += symbol
        column += 1 # Point to next column.

        # If there are no more columns OR we're at a shaded box, go back to
        # the first column and the next row:
        if (column == numOfColumns) or (column == numOfColumns - 1 and row >= numOfRows - numOfShadedBoxes):
            column = 0
            row += 1

    return ''.join(plaintext)

def test():
    assert decryptMessage([2, 4, 1, 5, 3], "IS HAUCREERNP F") == "CIPHERS ARE FUN"
    assert decryptMessage([2,4,6,8,10,1,3,5,7,9],"XOV EK HLYR NUCO HEEEWADCRETL CEEOACT KD") == "EXCELLENT WORK YOU HAVE CRACKED THE CODE"
    assert decryptMessage([1, 3, 2],"ADGCFBE") == "ABCDEFG"
    # print(decryptMessage([1,3,2], "ADGCFBE"))
    # print(decryptMessage([2, 4, 1, 5, 3], "IS HAUCREERNP F"))
    assert(decryptMessage([2, 1], "EL OLHLOWRD")) == "HELLO WORLD"
    # print (decryptMessage([2,4,6,8,10,1,3,5,7,9], "XOV EK HLYR NUCO HEEEWADCRETL CEEOACT KD"))

from sys import flags
import math

if __name__ == "__main__" and not flags.interactive:
    test()
