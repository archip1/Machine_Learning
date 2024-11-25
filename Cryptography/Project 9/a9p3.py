#!/usr/bin/env python3

# ---------------------------------------------------------------
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
# ---------------------------------------------------------------

"""
Assignment 9 Problem 3
"""

from sys import flags
from typing import List

SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.'

def blockSizeHack(blocks: List[int], n: int, e: int) -> str:
    """
    Hack RSA assuming a block size of 1
    """

    decryption = []

    # for each block, iterate thru each symbol, if formula is matched, add that symbol to decrypted text list
    for block in blocks:
        for i in range(len(SYMBOLS)):
            if block == pow(i, e, n): # if block equals the corresponding i (TEXTBOOK)
                decryption.append(SYMBOLS[i])

    return("".join(decryption))

def test():
    "Run tests"
    blocks = [2361958428825, 564784031984, 693733403745, 693733403745,2246930915779, 1969885380643]
    n = 3328101456763
    e = 1827871
    assert blockSizeHack(blocks, n, e) == "Hello."
    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking

# Invoke test() if called via `python3 a9p1.py`
# but not if `python3 -i a9p1.py` or `from a9p1 import *`
if __name__ == '__main__' and not flags.interactive:
    test()
