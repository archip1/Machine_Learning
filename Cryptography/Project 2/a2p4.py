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

def decryptMystery():

    wordList = "" # create an empty list to store all present words
    filename = "mystery.txt"

    file = open(filename, 'r')
    
    # content = fileObj.read()
    # If the input file does not exist, then the program terminates early:
    # if not os.path.exists(inputFilename):
    # print('The file %s does not exist. Quitting...' % (inputFilename))
    # sys.exit()

    #  Read file
    for line in file:
        # line = line.split()
        # print("type: ", type(line))
        # print("line: ", line)

        wordList += line
    file.close()

    return(decryptMessage([8,1,6,2,10,4,5,3,7,9], wordList))

def test():
    decriptedmsg = decryptMystery()

    # Wite to file
    file = open("mystery.dec.txt", 'w')
    file.write(decriptedmsg)
    file.close()

    # print("decrypted: ", decryptMystery())

from sys import flags
import math

if __name__ == "__main__" and not flags.interactive:
    test()
