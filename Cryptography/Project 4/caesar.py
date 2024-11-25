"""
CMPUT 331 Assignment 1 Student Solution
September 2023
Author: <Archi Patel>
"""

import string, util
from sys import flags

# testing purposes, creates the 52 letters
LETTERS = ''.join([u+l for u, l in 
    zip(string.ascii_uppercase, string.ascii_lowercase)])

# Create two dictionaries: the first mapping each letter to its corresponding shift
# second allowing the reverse mapping of shifts to their respective letters
shift_map = {}
letter_map = {}

def get_map(letters=LETTERS):
    # letters = input()
    number = 0
    for i in letters:
        shift_map[i] = number
        letter_map[number] = i
        number += 1
    # print("shift: ", shift_map, "\n")
    # print("letter: ", letter_map)

    return shift_map, letter_map

def caesar(key: str, message: str):
    global SHIFTDICT, LETTERDICT 
    SHIFTDICT, LETTERDICT = get_map()

    # message = "TCjBnMb Rm 2023 dzLi zKnTs 331!"
    # key = "XxYyZz"
    translated = ''
    count = 0
    # print(shiftamount)

    for i in range(len(message)):
        if message[i] in LETTERDICT.values():
            # Find the number for the letter and subtract the shift amount to it
            num = SHIFTDICT[message[i]]
            # print("before: ", num)

            # Ensure that when you run out of letters in the key, you wrap around
            keylen = count
            while keylen >= len(key):
                keylen -= len(key)

            shiftamount = SHIFTDICT[key[keylen]]
            num -= shiftamount
            # print("after: ", num)

            # If the map value is greater than or less that the number of letters in the dictionary,
            # in order to wrap around the letters we can add or subtract the length of the letters
            if num >= len(SHIFTDICT):
                num -= len(SHIFTDICT)
                # print("bw: ", num, "\n")
            elif num < 0:
                num += len(SHIFTDICT)

            # increment index of key
            count += 1
            translated += LETTERDICT[num]
            # print(LETTERDICT[num])
        else:
            translated = translated + message[i]
    
    return translated