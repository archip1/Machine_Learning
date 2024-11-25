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

"""
function: encryptMessage

Encrypts plain text with the rail fence cipher using the key as the number of rows

Arguements: 
    - key: an integer between 2 and the length of the message to determine how many rails are to be used
    - message: a string

Returns: 
    - the message encrypted with rail fence cipher

"""
def encryptMessage(key: int, message: str):
    ciphertext = ""

    # Each string in ciphertext represents a row in the grid.
    ciphertext = [''] * key

    # Keeps track of the row you want to append the letter to
    column = 0
    # Keeps track of if the rail fence is going up or down the rows
    direction = "down"

    # Loop through each letter in the plaintext and add the letter to each row depending on the direction.
    for letter in message:

        # if you reach the top row of the grid, toggle direction to go down 
        if column == 0:
            ciphertext[column] += letter
            column += 1
            direction = "down"

        # if you reach the top bottom of the grid, toggle direction to go up 
        elif column == key-1:
            ciphertext[column] += letter
            column -= 1
            direction = "up"

        elif direction == "down":
            ciphertext[column] += letter
            column += 1
            
        elif direction == "up":
            ciphertext[column] += letter
            column -= 1
        
    # Convert the ciphertext list into a single string value and return it.
    return ''.join(ciphertext)


"""
function: decryptMessage

Decrypts cipher text with the rail fence cipher using the key as the number of rows

Arguements: 
    - key: an integer between 2 and the length of the message to determine how many rails are to be used
    - message: a string

Returns: 
    - the message decrypted with rail fence cipher that reverses the encrypt function

"""
def decryptMessage(key: int, message: str):

    plaintext = ''
    # Each string in ciphertext represents a column in the grid. The grid is numof letters in the msg wide, and num of rows, tall
    plaintext = [''] * len(message)

    # keeps track of the letter we are inserting into the grid
    letter = 0

    # loop thru the grid and every placement of a letter and if it matches the i'th then append the letter
    # you are essentially appending the letters for each row and iterating thru all the rows
    for i in range(key):

        # keep trak of row, column, and dircation of iteration
        column = 0
        row = 0
        direction = "down"

        for j in range(len(message)):

            # while iterating thru the rail cipher path, if row = the row we are filling out, append the letter
            if row == i:
                plaintext[column] += message[letter]
                letter += 1

            # to keep iterating when u reach the top of the table, toggle direction to down
            if row == 0:
                row += 1
                direction = "down"

            # to keep iterating when u reach the bottom of the table, toggle direction to up
            elif row == key-1:
                row -= 1
                direction = "up"

            elif direction == "down":
                row += 1

            elif direction == "up":
                row -= 1

            column += 1 
            
    # Convert the ciphertext list into a single string value and return it.
    return ''.join(plaintext)

def test():
    assert decryptMessage(2, encryptMessage(2, "SECRET")) == "SECRET"
    assert decryptMessage(3, encryptMessage(3, "CIPHERS ARE FUN")) == "CIPHERS ARE FUN"
    assert decryptMessage(4, encryptMessage(4, "HELLO WORLD")) == "HELLO WORLD"
    # a = encryptMessage(2, "SECRET")
    # b = encryptMessage(3, "CIPHERS ARE FUN")
    # c = encryptMessage(4, "HELLO WORLD")
    # print (a)
    # print (b)
    # print (c)    
    # e = decryptMessage(2, "SCEERT")
    # f = decryptMessage(3, "CEAFIHR R UPSEN")
    # g = decryptMessage(4, "HWE OLORDLL")
    # print (e)
    # print (f)
    # print (g)


from sys import flags
import math

if __name__ == "__main__" and not flags.interactive:
    test()
