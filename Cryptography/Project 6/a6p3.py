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
Problem 3
"""

from sys import flags
import a6p2

def bestSuccessor(mapping: dict, ciphertext: str, frequencies: dict, n: int) -> dict:
    LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # initoial key score
    bestKeyScore = a6p2.keyScore(mapping, ciphertext, frequencies, n)
    bestMapping = mapping.copy()

    # if space in mapping dont swap space 
    space = " "
    if space in bestMapping:
        iter = len(bestMapping) - 1
    else:
        iter = len(bestMapping)

    # get all keys
    keys = list(bestMapping.keys())

    # for each index, iterate thru and swap 
    for key1 in range(iter): # 0 to 26 (number of char in LETTERS)
        for key2 in range(key1+1, iter): # x to 26 (currectkey, x, to number of char in LETTERS)
            
            # Get the char at the currect LETTER position (eg. C)
            keyONE = keys[key1]
            keyTWO = keys[key2]

            TempMapping = mapping.copy()
            # get key values
            value1 = TempMapping[keyONE]
            value2 = TempMapping[keyTWO]

            # Swap values
            TempMapping[keyONE] = value2
            TempMapping[keyTWO] = value1

            # After swapping the values, get it's score
            score = a6p2.keyScore(TempMapping, ciphertext, frequencies, n)

            if score == bestKeyScore: # if the new score is equal to the best score,
                # Use the function given to us to break ties and find out the best mapping
                bestMapping = breakKeyScoreTie(mapping, TempMapping, bestMapping)
            
            elif score > bestKeyScore: # if the new score is better than the currect best,
                bestKeyScore = score # new best score is obtained.
                bestMapping = TempMapping # Thus, currect mapping is new best mapping.
    
    # In the end, return the best mapping
    return(bestMapping)

def breakKeyScoreTie(originalMapping, successorMappingA, successorMappingB):
    """
    Break the tie between two successor mappings that have the same keyscore

    originalMapping: mapping the the other parameters are successors to
    successorMappingA: mapping that has had two keys swapped
    successorMappingB: mapping that has had two other keys swapped

    Example usage:
    originalMapping = {"A": "A", "B": "B", "C": "C"}
    # Mapping with B and C switched
    successorMappingA = {"A": "A", "B": "C", "C": "B"}
    # Mapping with A and C switched
    successorMappingB = {"A": "C", "B": "B", "C": "A"}

    # AC < BC so this function will return successorMappingB
    assert breakKeyScoreTie(originalMapping, successorMappingA, successorMappingB) == successorMappingB
    """
    aSwapped = "".join(sorted(k for k, v in (
        set(successorMappingA.items()) - set(originalMapping.items()))))
    bSwapped = "".join(sorted(k for k, v in (
        set(successorMappingB.items()) - set(originalMapping.items()))))
    return successorMappingA if aSwapped < bSwapped else successorMappingB

def test():
    "Run tests"
    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking
    assert breakKeyScoreTie({"A": "A", "B": "B", "C": "C"}, {"A": "A", "B": "C", "C": "B"}, {
                            "A": "C", "B": "B", "C": "A"}) == {"A": "C", "B": "B", "C": "A"}
    assert breakKeyScoreTie({"A": "A", "B": "B", "C": "C", "D": "D"}, {
                            "A": "B", "B": "A", "C": "C", "D": "D"}, {"A": "A", "B": "B", "C": "D", "D": "C"}) == {"A": "B", "B": "A", "C": "C", "D": "D"}

if __name__ == "__main__" and not flags.interactive:
    test()