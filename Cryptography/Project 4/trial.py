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

    # matching_words = match_word_to_dictionary_word(blank_matches, dictionaryWords, plain_words)

    for blank_word in blank_matches:

        blank_word_upper = blank_word.upper()
        pattern = re.sub(r'[^A-Z_]', '', blank_word_upper)
        find = re.compile(pattern.replace('_', r'[A-Z]'))

        possible_words = [word for word in dictionaryWords if re.findall(find, word)]

        possible_length_words = [word for word in possible_words if len(word) == len(pattern)]

        if len(possible_length_words) == 1:

            word = possible_words[0]

            # get replacement letter
            index = plain_words.index(blank_word)
            encrypted_word = encrypted_words[index]

            for i in range(len(encrypted_word)):
                if blank_word[i] == "_" and len(intersected_map[(encrypted_word[i]).upper()]) > 1:
                    intersected_map[(encrypted_word[i]).upper()] = [word[i]]

                    plain_text_blanks = simpleSubHacker.decryptWithCipherletterMapping(message, intersected_map)
                    plain_words = plain_text_blanks.split()     

                    for j in multiple_choice_letters:
                        if j != encrypted_word[i].upper() and word[i].upper() in intersected_map[j]:
                                intersected_map[j].remove(word[i].upper())

                    multiple_choice_letters.remove(encrypted_word[i].upper())



    return plain_text_blanks

def test():
    # Provided test.
    message = 'Sy l nlx sr pyyacao l ylwj eiswi upar lulsxrj isr sxrjsxwjr, ia esmm rwctjsxsza sj wmpramh, lxo txmarr jia aqsoaxwa sr pqaceiamnsxu, ia esmm caytra jp famsaqa sj. Sy, px jia pjiac ilxo, ia sr pyyacao rpnajisxu eiswi lyypcor l calrpx ypc lwjsxu sx lwwpcolxwa jp isr sxrjsxwjr, ia esmm lwwabj sj aqax px jia rmsuijarj aqsoaxwa. Jia pcsusx py nhjir sr agbmlsxao sx jisr elh. -Facjclxo Ctrramm'
    print(hackSimpleSub(message))

    message2 = "The rune of death was stolen and the demigods began to fall, starting with Godwyn the Golden. Queen Marika was driven to the brink. The Shattering ensued; a war that wrought only darkness. The Elden Ring was broken, but by whom? And why? What could the Demigods ever hope to win by warring? The conqueror of the stars, General Radahn. And the Blade of Miquella, Malenia the Severed. These two were the mightiest to remain, and locked horns in combat. But there would be no victor. And so, we inhabit a fractured world. Awaiting the arrival of the Elden Lord. Unless of course, thou shouldst take the crown?"
    print(hackSimpleSub(message2))

    # End of provided test.
    

if __name__ == '__main__':
    test()
