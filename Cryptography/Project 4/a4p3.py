#!/usr/bin/env python3

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
Enhanced substitution cipher solver.
"""

import re, simpleSubHacker

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def load_dictionary():
    # open the dictionary text file, read line by line and store it in a list
    dictionaryWords = []
    filename = "dictionary.txt"
    file = open(filename, 'r')
    for line in file:
        line = line[:-1]
        dictionaryWords.append(line)
    file.close()

    return dictionaryWords

def get_letters_with_multiple_translations(intersected_map):
    # Get all letters in cipher text that have more than one possible translation to plaintext
    multiple_choice_letters = []

    for i in intersected_map:
        if len(intersected_map[i]) > 1:
            multiple_choice_letters.append(i)

    return multiple_choice_letters

def get_blank_words(plain_words):
    # find all words with _ in the word using regex by iterating thru the list of words of the plain text
    blank_matches = []
    for word in plain_words:
        pattern = re.compile(r'\w*_\w*') # 0+ words charachters, _, 0+ words charachters
        if re.search(pattern, word):
            blank_matches.append(word)

    return blank_matches

def hackSimpleSub(message: str):
    """
    Simple substitution cipher hacker.
    First runs the textbook program to get an initial, potentially incomplete decipherment.
    """

    # load dictionary to a list
    dictionaryWords = load_dictionary()

    # create intersected map and get the plan text with blanks
    intersected_map = simpleSubHacker.hackSimpleSub(message)
    plain_text_blanks = simpleSubHacker.decryptWithCipherletterMapping(message, intersected_map)

    # Get all letters in cipher text that have more than one possible translation to plaintext
    multiple_choice_letters = get_letters_with_multiple_translations(intersected_map)

    # create lists of both cipher text and plain text
    plain_words = plain_text_blanks.split()
    encrypted_words = message.split()

    # get all words with blanks
    blank_matches = get_blank_words(plain_words)

    # matching_words = []

    # iterate thru the words with blanks
    while len(blank_matches) > 0:
        blank_match = blank_matches.pop()
        blank_match_upper = blank_match.upper()

        # match blank word with a dictionary word
        pattern = re.sub(r'[^A-Z_]', '', blank_match_upper)
        find = re.compile(pattern.replace('_', r'[A-Z]'))
        for word in dictionaryWords:
            match = re.findall(find, word)

            # there is a match and the length of the match is = to the length of the word and the word still has a blank
            if match and len(match[0]) == len(pattern) and match[0] and blank_match in plain_words:
                
                # find the index of the blank word in list of plain words and find the encrypted word at that index
                index = plain_words.index(blank_match)
                encrypted_word = encrypted_words[index]

                # find the index of the blank and the enrcypted letter at that index
                index_blank = blank_match.index("_")
                position = encrypted_word[index_blank]

                # 
                replacement_letter = match[0][index_blank]
                if replacement_letter in intersected_map[position.upper()]:
                    intersected_map[position.upper()] = [replacement_letter]
                
                    plain_text_blanks = simpleSubHacker.decryptWithCipherletterMapping(message, intersected_map)
                    plain_words = plain_text_blanks.split()

                    for i in multiple_choice_letters:
                        if i != position.upper():
                            if replacement_letter.upper() in intersected_map[i]:
                                intersected_map[i].remove(replacement_letter.upper())

                    multiple_choice_letters.remove(position.upper())

    return plain_text_blanks

def test():
    # Provided test.
    message = 'Sy l nlx sr pyyacao l ylwj eiswi upar lulsxrj isr sxrjsxwjr, ia esmm rwctjsxsza sj wmpramh, lxo txmarr jia aqsoaxwa sr pqaceiamnsxu, ia esmm caytra jp famsaqa sj. Sy, px jia pjiac ilxo, ia sr pyyacao rpnajisxu eiswi lyypcor l calrpx ypc lwjsxu sx lwwpcolxwa jp isr sxrjsxwjr, ia esmm lwwabj sj aqax px jia rmsuijarj aqsoaxwa. Jia pcsusx py nhjir sr agbmlsxao sx jisr elh. -Facjclxo Ctrramm'
    print(hackSimpleSub(message))

    # message2 = "General relativity, also known as the general theory of relativity and Einstein's theory of gravity, is the geometric theory of gravitation published by Albert Einstein in 1915 and is the current description of gravitation in modern physics. General relativity generalises special relativity and refines Newton's law of universal gravitation, providing a unified description of gravity as a geometric property of space and time or four-dimensional spacetime."
    # print(hackSimpleSub(message2))

    # End of provided test.
    

if __name__ == '__main__':
    test()
