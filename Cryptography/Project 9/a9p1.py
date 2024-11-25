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
Assignment 9 Problem 1
"""

from sys import flags
from typing import Tuple
import primeNum as prime

#  EXTRA CODE

# def is_prime(number):
#     for i in range(2, int(number**0.5) + 1):
#         if number % i == 0:
#             return False
#     return True

# # # FROM TEXTBOOK
# def gcd(a, b):
#     # Return the Greatest Common Divisor of a and b using Euclid's Algorithm
#     while a != 0:
#         a, b = b % a, a
#     return b

# # FROM TEXTBOOK
# def findModInverse(a, m):
#     # Return the modular inverse of a % m, which is
#     # the number x such that a*x % m = 1

#     if gcd(a, m) != 1:
#         return None # No mod inverse exists if a & m aren't relatively prime.

#     # Calculate using the Extended Euclidean Algorithm:
#     u1, u2, u3 = 1, 0, a
#     v1, v2, v3 = 0, 1, m
#     while v3 != 0:
#         q = u3 // v3 # Note that // is the integer division operator
#         v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3
#     return u1 % m

def finitePrimeHack(t: int, n: int, e: int) -> Tuple[int, int, int]:
    """
    Hack RSA assuming there are no primes larger than t
    """
    # iterate to threshold and see if the number is prime and if its other factor to multiply to make is n is also prime then 
    for i in range(2, t+1):
        if prime.isPrime(i):
            p = int(i)
            q = n // p
            if p * q == n and prime.isPrime(q):
                q = int(q)

                # calculate phi and use the formula given in the instructions to get d
                phi = (p - 1) * (q - 1)
                d = int(pow(e, -1, phi))

                if p < q:
                    return (p, q, d)
                else:
                    return (q, p, d)

                # ANOTHER APPROACH - THE OTHER IS EASIER
                # if gcd(e, phi) == 1:
                #     d = findModInverse(e, phi)
                #     if p < q:
                #         # print((int(p), int(q), int(d)))
                #         return (p, q, d)
                #     else:
                #         # print((int(q), int(p), int(d)))
                #         return (q, p, d)            

def test():
    "Run tests"
    # print(gcd(80,20))
    # print(findModInverse(8, 11))
    # print(prime.isPrime(786241))

    assert finitePrimeHack(100, 493, 5) == (17, 29, 269)
    assert finitePrimeHack(2**16,2604135181,1451556085) == (48533, 53657, 60765)
    assert finitePrimeHack(2**20,584350822261,567347) ==  (743221, 786241, 454279775483)

    print(finitePrimeHack(2**14,29177,293))
    # print(finitePrimeHack(2**14,160728233,8951))
    # print(finitePrimeHack(2**14,106646587,14023))
    # print(finitePrimeHack(2**14,255706897,9061))
    # print(finitePrimeHack(2**14,167131319,8215))
    # print(finitePrimeHack(2**14,186609961,11283))

    # TODO: test thoroughly by writing your own regression tests
    # This function is ignored in our marking


# Invoke test() if called via `python3 a9p1.py`
# but not if `python3 -i a9p1.py` or `from a9p1 import *`
if __name__ == '__main__' and not flags.interactive:
    test()
