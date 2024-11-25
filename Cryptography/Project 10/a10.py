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
Assignment 10
"""
from sys import flags
from collections import Counter

def compute_SSD (text: str) -> dict[int, float]:
    text = ''.join(ch for ch in text if ch.isalpha())
    text = text.upper()

    # count freq
    freq = Counter(text)

    # Get relative frequency
    SSD = []
    for count in freq.values():
        SSD.append(count/len(text))

    # Return sorted dictionary
    SSD = sorted(SSD, reverse=True)
    return SSD

def sum_of_squared_differences_ssd(P1: list[float], P2: list[float]) -> float:
    # Extend the shorter list with zeros to make the lists 0
    if len(P1) < len(P2):
        P1 += [0] * (len(P2) - len(P1))

    elif len(P2) < len(P1):
        P2 += [0] * (len(P1) - len(P2))

    # Compute the sum of squared differences
    sum_pow = 0
    for i in range(len(P1)):
        difference = P1[i] - P2[i]
        pow = difference ** 2
        sum_pow += pow
    return sum_pow

def cliSSD(ciphertext: str, files):
    """
    Args:
        ciphertext (str)
        files (list of str)
    Returns:
        dict"""
    
    # get ssd of ciphertext
    SSD_ciphertext = compute_SSD(ciphertext)
    distances = {}

    # iterate through the files
    for file in files:
        f = open(file, encoding='UTF-8', errors="ignore")
        text = f.read()
        f.close()

        # get ssd of text language
        SSD_text = compute_SSD(text)

        # calculate the differnces in ssd
        distance = sum_of_squared_differences_ssd(SSD_ciphertext, SSD_text)
        distances[file] = distance

    # return distances as a dictionary
    return(distances)

def compute_dpd (text: str):

    text = text.upper()
    words = text.split()
    # text = ''.join(ch for ch in text if ch.isalpha())

    # find the decomposition pattern for each word
    patterns = []
    for word in words:
        pattern = Counter(word)
        patterns.append(tuple(sorted(pattern.values(), reverse=True))    )

    # the frequency of each pattern
    freq = Counter(patterns)

    # the relative frequency
    DPD = {}
    for pattern, count in freq.items():
        DPD[pattern] =  count / len(words)

    return (DPD)

def sum_of_squared_differences_dpd(DPD_ciphertext, DPD_text):
    distance = 0
    # combine all unique keys, chat gpt
    all = set(DPD_ciphertext.keys()) | set(DPD_text.keys())

    # find differences in all patterns
    for pattern in all:
        if pattern not in DPD_text.keys():
            distance += (DPD_ciphertext[pattern]) ** 2
        elif pattern not in DPD_ciphertext.keys():
            distance += (DPD_text[pattern]) ** 2
        else:
            distance += (DPD_ciphertext[pattern] - DPD_text[pattern]) ** 2

    # return sum of differences
    return distance

def cliDPD(ciphertext: str, files):
    """
    Args:
        ciphertext (str)
        files (list of str)
    Returns:
        dict
    """
    # get dpd of ciphertext
    DPD_ciphertext = compute_dpd(ciphertext)
    distances = {}

    # iterate through the files
    for file in files:
        f = open(file, encoding='UTF-8', errors="ignore")
        text = f.read()
        f.close()

        # get dpd of text language
        DPD_text = compute_dpd(text)

        # calculate the differnces in dpd
        distance = sum_of_squared_differences_dpd(DPD_ciphertext, DPD_text)
        distances[file] = distance

    # return distances as a dictionary
    return(distances)

def cliSSDTest(ciphertext_files, sampletext_files):
    """
    Args:
        ciphertext_files (list of str)
        sampletext_files (list of str)
    Returns:
        dict
    """
    cipher_sample = {}

    # iterate through each ciphertext file and open it
    for file in ciphertext_files:
        f = open(file, encoding='UTF-8', errors="ignore")
        cipher = f.read()
        f.close()

        # call calculate ssd funct, find the minimumn sample file language and store that in a dictionary
        temp_dict = cliSSD(cipher, sampletext_files)
        min_key = min(temp_dict, key=lambda k: temp_dict[k])
        file = file.split('/')[-1]
        min_key = min_key.split('/')[-1]
        
        cipher_sample[file] = min_key

    # return a dictionary with each cipherfile and value as the language detected
    # print(cipher_sample)
    return (cipher_sample)

def cliDPDTest(ciphertext_files, sampletext_files):
    """
    Args:
        ciphertext_files (list of str)
        sampletext_files (list of str)
    Returns:
        dict
    """
    cipher_sample = {}

    # iterate through each ciphertext file and open it
    for file in ciphertext_files:
        f = open(file, encoding='UTF-8', errors="ignore")
        cipher = f.read()
        f.close()

        # call calculate dpd funct, find the minimumn sample file language and store that in a dictionary
        temp_dict = cliDPD(cipher, sampletext_files)
        min_key = min(temp_dict, key=lambda k: temp_dict[k])

        file = file.split('/')[-1]
        min_key = min_key.split('/')[-1]

        cipher_sample[file] = min_key

    # return a dictionary with each cipherfile and value as the language detected
    # print(cipher_sample)
    return (cipher_sample)

def test():
    # ciphertextfiles = ["texts/ciphertext_bg_1.txt", "texts/ciphertext_bg_2.txt", "texts/ciphertext_bg_3.txt", "texts/ciphertext_bg_4.txt", "texts/ciphertext_bg_5.txt", "texts/ciphertext_bg_6.txt", "texts/ciphertext_bg_7.txt", "texts/ciphertext_bg_8.txt", "texts/ciphertext_bg_9.txt", "texts/ciphertext_de_1.txt", "texts/ciphertext_de_2.txt", "texts/ciphertext_de_3.txt", "texts/ciphertext_de_4.txt", "texts/ciphertext_de_5.txt", "texts/ciphertext_de_6.txt", "texts/ciphertext_de_7.txt", "texts/ciphertext_de_8.txt", "texts/ciphertext_de_9.txt", "texts/ciphertext_el_1.txt", "texts/ciphertext_el_2.txt", "texts/ciphertext_el_3.txt", "texts/ciphertext_el_4.txt", "texts/ciphertext_el_5.txt", "texts/ciphertext_el_6.txt", "texts/ciphertext_el_7.txt", "texts/ciphertext_el_8.txt", "texts/ciphertext_el_9.txt", "texts/ciphertext_en_1.txt", "texts/ciphertext_en_2.txt", "texts/ciphertext_en_3.txt", "texts/ciphertext_en_4.txt", "texts/ciphertext_en_5.txt", "texts/ciphertext_en_6.txt", "texts/ciphertext_en_7.txt", "texts/ciphertext_en_8.txt", "texts/ciphertext_en_9.txt", "texts/ciphertext_es_1.txt", "texts/ciphertext_es_2.txt", "texts/ciphertext_es_3.txt", "texts/ciphertext_es_4.txt", "texts/ciphertext_es_5.txt", "texts/ciphertext_es_6.txt", "texts/ciphertext_es_7.txt", "texts/ciphertext_es_8.txt", "texts/ciphertext_es_9.txt", "texts/ciphertext_fr_1.txt", "texts/ciphertext_fr_2.txt", "texts/ciphertext_fr_3.txt", "texts/ciphertext_fr_4.txt", "texts/ciphertext_fr_5.txt", "texts/ciphertext_fr_6.txt", "texts/ciphertext_fr_7.txt", "texts/ciphertext_fr_8.txt", "texts/ciphertext_fr_9.txt", "texts/ciphertext_it_1.txt", "texts/ciphertext_it_2.txt", "texts/ciphertext_it_3.txt", "texts/ciphertext_it_4.txt", "texts/ciphertext_it_5.txt", "texts/ciphertext_it_6.txt", "texts/ciphertext_it_7.txt", "texts/ciphertext_it_8.txt", "texts/ciphertext_it_9.txt", "texts/ciphertext_nl_1.txt", "texts/ciphertext_nl_2.txt", "texts/ciphertext_nl_3.txt", "texts/ciphertext_nl_4.txt", "texts/ciphertext_nl_5.txt", "texts/ciphertext_nl_6.txt", "texts/ciphertext_nl_7.txt", "texts/ciphertext_nl_8.txt", "texts/ciphertext_nl_9.txt", "texts/ciphertext_pl_1.txt", "texts/ciphertext_pl_2.txt", "texts/ciphertext_pl_3.txt", "texts/ciphertext_pl_4.txt", "texts/ciphertext_pl_5.txt", "texts/ciphertext_pl_6.txt", "texts/ciphertext_pl_7.txt", "texts/ciphertext_pl_8.txt", "texts/ciphertext_pl_9.txt", "texts/ciphertext_ru_1.txt", "texts/ciphertext_ru_2.txt", "texts/ciphertext_ru_3.txt", "texts/ciphertext_ru_4.txt", "texts/ciphertext_ru_5.txt", "texts/ciphertext_ru_6.txt", "texts/ciphertext_ru_7.txt", "texts/ciphertext_ru_8.txt", "texts/ciphertext_ru_9.txt"]

    # samplefiles = ["texts/sample_bg.txt", "texts/sample_de.txt", "texts/sample_el.txt", "texts/sample_en.txt", "texts/sample_es.txt", "texts/sample_fr.txt", "texts/sample_it.txt", "texts/sample_nl.txt", "texts/sample_pl.txt", "texts/sample_ru.txt"]

    # a = (cliSSD(ciphertext, samplefiles))
    # for k, v in a.items():
    #     print (k, v)

    # a = (cliDPD(ciphertext, samplefiles))
    # for k, v in a.items():
    #     print (k, v)

    # a = cliSSDTest(ciphertextfiles, samplefiles)
    # for k, v in a.items():
    #     print (k, v)

    # a = cliDPDTest(ciphertextfiles, samplefiles)
    # for k, v in a.items():
    #     print (k, v)
    pass

if __name__ == "__main__" and not flags.interactive:
    test()