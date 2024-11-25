#!/usr/bin/python3

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
CMPUT 331 Assignment 1 Student Solution
September 2023
Author: <Archi Patel>
"""


import string
from sys import flags

# testing purposes, creates the 52 letters 
LETTERS = ''.join([u+l for u, l in 
    zip(string.ascii_uppercase, string.ascii_lowercase)])

# Create two dictionaries: the first mapping each letter to its corresponding shift
# second allowing the reverse mapping of shifts to their respective letters
shift_map = {}
letter_map = {}

"""
function: get map

Establishes the connections between letters and their respective shift amounts

Arguements: letters - 52 letters that will create the dictionaries

Returns: Two dictionaries: one that maps letters to it's shifts and second, reversemaps

"""
def get_map(letters=LETTERS):
    # letters = input()
    number = 0
    # For each letter, add the letter as a key in one dictionary and then value in other and 
    # then the shift number as value for one and key for the next
    for i in letters:
        shift_map[i] = number
        letter_map[number] = i
        number += 1
    # print("shift: ", SHIFTDICT, "\n")
    # print("letter: ", LETTERDICT)

    return shift_map, letter_map

"""
function: encrypt

return the message string such that each letter has been caesar- shifted by the key amount. Non-letters characters (such as spaces or punctuation) are to be appended to the resulting string unmodified

Arguements: Message - the plaintext we want to encrypt
key: the number of letters we want to shift by 

Returns: The encrypted text

"""
def encrypt(message: str, key: str):
    # message = "Welcome to 2023 Fall CMPUT 331!"
    # key = "x"

    # Store the new string
    translated = ''

    # Find the shift value from the key
    shiftamount = SHIFTDICT[key]

    # print(shiftamount)

    # Iterate through the message, if its a letter in the dictionary, shift or else append to string
    for symbol in message:
        if symbol in LETTERDICT.values():
            # Find the number for the letter and add the shift amount to it
            num = SHIFTDICT[symbol]
            # print("before: ", num)
            num += shiftamount
            # print("after: ", num)

            # If the map value is greater than or less that the number of letters in the dictionary,
            # in order to wrap around the letters we can add or subtract the length of the letters
            if num >= len(SHIFTDICT):
                num -= len(SHIFTDICT)
                # print("bw: ", num, "\n")
            elif num < 0:
                num += len(SHIFTDICT)

            translated += LETTERDICT[num]
            # print(LETTERDICT[num])
        else:
            translated = translated + symbol
    
    return translated

""" 
function: decrypt

reverse the shift performed by encrypt and return the original plaintext

Arguements: Message - the cyphertext we want to decrypt
key: the number of letters we want to shift by 

Returns: The decrypted text
"""
def decrypt(message: str, key: str):
    # message = "tCJAMKC RM 2023 cYJJ zjmrq 331!"
    # key = "x"

    # Store the new string
    translated = ''

    # Find the shift value from the key
    shiftamount = SHIFTDICT[key]

    # Iterate through the message, if its a letter in the dictionary, shift or else append to string
    for symbol in message:
        if symbol in LETTERDICT.values():
            # Find the number for the letter and subtract the shift amount to it, to reverse it
            num = SHIFTDICT[symbol]
            num -= shiftamount

            # If the map value is greater than or less that the number of letters in the dictionary,
            # in order to wrap around the letters we can add or subtract the length of the letters
            if num >= len(SHIFTDICT):
                num -= len(SHIFTDICT)
            elif num < 0:
                num += len(SHIFTDICT)

            translated += LETTERDICT[num]

        else:
            translated = translated + symbol

    return translated

def test():
    global SHIFTDICT, LETTERDICT 
    SHIFTDICT, LETTERDICT = get_map()
    assert decrypt(encrypt("foo", "g"), "g") == "foo"
    # for i in SHIFTDICT:
    #     print(i, " ", decrypt("URYYB JBEYQ", i))

    # e = encrypt("Welcome to 2023 Fall CMPUT 331!", "x")
    # d = decrypt("tCJAMKC RM 2023 cYJJ zjmrq 331!", "x")
    # print("e: ", e, "\n", "d: ", d)

if __name__ == "__main__" and not flags.interactive:
    test()