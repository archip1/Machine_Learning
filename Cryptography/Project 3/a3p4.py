#!/usr/bin/python3

#---------------------------------------------------------------
#
# CMPUT 331 Student Submission License
# Version 1.0
# Copyright 2023 <<Archi Patel>>
#
# redistribution is forbidden in all circumstances. Use of this software
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
# the UrL or other repository locating information, to the following email
# address:
#
#          gkondrak <at> ualberta.ca
#
#---------------------------------------------------------------

"""
CMPUT 331 Assignment 3 Student Solution
September 2023
Author: <Archi Patel>
"""

def crack_rng(m, sequence):
    r2, r3, r4, r5, r6 = tuple(sequence)

    # THREE EQNS
    # r6 = a*r5 + b*r4 + c
    # r5 = a*r4 + b*r3 + c
    # r4 = a*r3 + b*r2 + c

    # SUBTRACT eqn 1 - 2 and 2 and 3:
    # q^-1 * r6 - r5 = q^-1 * a*(r5-r4) + q^-1 * b*(r4-r3)
    # p^-1 * r5 - r4 = p^-1 * a*(r4-r3) + p^-1 * b*(r3-r2)

    # q_inv * r = q_inv * a*q + q_inv * b*p
    # p_inv * q = p_inv * a*p + p_inv * b*s    
    r = r6 - r5
    q = r5 - r4
    p = r4 - r3
    s = r3 - r2


    # q_inv * r = a + q_inv * b*p
    # p_inv * q = a + p_inv * b*s
    # q_inv = findModInverse(q, m)
    # p_inv = findModInverse(p, m)

    q_inv = pow(q, -1, m)
    p_inv = pow(p, -1, m)

    # Subtract the two eqn:
    # q_inv * r - p_inv * q = q_inv * b*p - p_inv * b*s
    # beta = b (q_inv*p - p_inv*s)
    # beta = b (alpha)
    beta = q_inv * r - p_inv * q
    alpha = q_inv*p - p_inv*s

    # MULTIPLY BY INVERSE
    # b = beta * alpha_inv
    # alpha_inv = findModInverse(alpha_inv, m)
    alpha_inv = pow(alpha, -1, m)
    b = beta * alpha_inv

    # mod m
    # b * (mod m)
    b = b % m

    # THREE EQNS
    # r6 = a*r5 + b*r4 + c
    # r5 = a*r4 + b*r3 + c
    # r4 = a*r3 + b*r2 + c

    # r = a*q + b*p
    a = q_inv * (r - b*p)
    a = a % m

    c = r4 - a*r3 - b*r2
    c = c % m

    return [a, b, c]

def test():

    assert crack_rng(17, [14, 13, 16, 3, 13]) == [3, 5, 9]
    assert crack_rng(9672485827, [4674207334, 3722211255, 3589660660, 1628254817, 8758883504]) == [22695477, 77557187, 259336153]
    assert crack_rng(101, [0, 91, 84, 16, 7]) == [29, 37, 71]
    assert crack_rng(222334565193649,[438447297,50289200612813,17962583104439,47361932650166,159841610077391]) == [1128889, 1023, 511]

    # try:
    #     assert crack_rng(17, [14, 13, 16, 3, 13]) == [3, 5, 9]
    #     assert crack_rng(9672485827, [4674207334, 3722211255, 3589660660, 1628254817, 8758883504]) == [22695477, 77557187, 259336153]
    #     assert crack_rng(101, [0, 91, 84, 16, 7]) == [29, 37, 71]
    #     assert crack_rng(222334565193649,[438447297,50289200612813,17962583104439,47361932650166,159841610077391]) == [1128889, 1023, 511]
    #     print("All test PASSED") 
    # except:
    #     print("One or more test FAILING")

    # print(crack_rng(17, [14, 13, 16, 3, 13]))
    # print(crack_rng(9672485827, [4674207334, 3722211255, 3589660660, 1628254817, 8758883504]))
    # print(crack_rng(101, [0, 91, 84, 16, 7]))
    # print(crack_rng(222334565193649,[438447297,50289200612813,17962583104439,47361932650166,159841610077391]))
    # print(crack_rng(467, [28, 137, 41, 118, 105]))

from sys import flags

if __name__ == "__main__" and not flags.interactive:
    test()