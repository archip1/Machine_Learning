#!/usr/bin/env python3

# ---------------------------------------------------------------
#
# CMPUT 331 Student Submission License
# Version 1.1
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
Assignment 8 Problems 1, 2 and 3
"""
from sys import flags
import re
import a7p234
import itertools

# English letter frequencies for calculating IMC (by precentage)
ENG_LETT_FREQ = {'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33, 'H': 6.09, 
                 'R': 5.99,  'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 
                 'G': 2.02,  'Y': 1.97, 'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 
                 'Q': 0.10,  'Z': 0.07}

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# From Assignment 6 Part 2
def keyScore(text: str, frequencies: dict, n: int ) -> float:

    ngram_score = 0.0

    # iterate thru each letter and get its relative frequency and add it to the n_gram score variable
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        if ngram in frequencies:
            ngram_score += frequencies[ngram]

    return(ngram_score)

def ngramsFreqsFromFile(textFile, n: int) -> dict:

    #  Read file and append letters to a list
    file = open(textFile, 'r')
    text = (file.read()).upper()
    file.close()

    textList = []
    for i in text:
        textList.append(i)
    
    ngram = []
    # create a list with letters as many times as the n gram
    loop = (len(textList)-n+1)
    for i in range(loop):
        for j in range(n):
            try:
                ngram.append(textList[j+i]) # Append the current ngram into the nGram list. 
            except:
                continue
    
    # Join the ngram characters into one single ngram
    i = 0
    for j in range(int(len(ngram)/n)):
        # Below is the link that was used for this:
        # https://www.geeksforgeeks.org/python-merge-list-elements/
        ngram[i : n+i] = [''.join(ngram[i : n+i])]
        i = i+1

    # Create a the Dictionary with frequency of n grams
    ngramDict={}
    for item in ngram:
        if item in ngramDict:
            value = ((ngramDict.get(item))*len(ngram)) # Find out the percent of time it occurs in ngramDict
            newValue = (value+1)/len(ngram)
            ngramDict[item] = newValue # Update the dict
        else:
            # if the ngram is not in the dict, put it in.
            value = (1/len(ngram))  
            ngramDict[item] = value

    return(ngramDict)

def getLetterFrequency(message):
    # Returns a dictionary of letter frequencies in the message
    # Divide each letter count by total number of letters in the message to get it's frequency
    letterCount = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 
                   'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 
                   'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 
                   'Y': 0, 'Z': 0}
    
    # iterate through the message and count occurences of each letter
    # count the number of letters in the length variable 
    length = 0
    for i in message:
        if i.upper() in letterCount:
            letterCount[i.upper()] += 1
            length += 1

    # divide each occurence by the total number of letters to get the frequency
    for i in letterCount:
        count = letterCount[i]
        letterCount[i] = count/length

    return letterCount

def getSubsequences(ciphertext, keylen):
    # This function takes in a ciphertext as a string and a key length as a int for its parameters
    # This function will return list of lists containing the characters in each subsequence
    subsequences_list = []
    subsequences = [''] * keylen

    # create subsequences
    position = 0
    for i in ciphertext:
        if position == keylen:
            position = 0

        subsequences[position] += i.upper()
        position += 1

    # make them all into lists
    for i in subsequences:
        temp = list(i)
        subsequences_list.append(temp)    

    return subsequences_list

def calculateTopIMC(subsequence):
    # Given a string, this function will calculate and return a list containing all 26 keys and their IMC values
    # Return a list of tuples containing key, IMC pairs from largest IMC to smallest

    freq = getLetterFrequency(subsequence)

    MaxScoreDict = {} 
    for i in range(26):
        IMC = 0

        for j in range(26):
            # Use the given formula to get IMC
            # Shift freq by 1 each loop and keep ENG_LETT_FREQ_List the same                      
            IMC = IMC + ((ENG_LETT_FREQ[LETTERS[j]]) * (freq[LETTERS[(i+j)%26]]))

        MaxScoreDict[LETTERS[i]] = IMC

    # for i in LETTERS:
    #     decryption = decryptVigenere(subsequence, i)
    #     freq = getLetterFrequency(decryption)
    #     imc = 0
    #     for j in decryption:
    #         imc += freq[j] * ENG_LETT_FREQ[j]
    #     MaxScoreDict[i] = imc

    sorted_dict = dict(sorted(MaxScoreDict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)
    return sorted_dict

def decryptVigenere(ciphertext, key):
    # This function takes in a vigenere ciphertext and it's key as the parameters
    # The decrypted message will be returned
    
    # Copied from the textbook modules
    decryption = ''

    keyIndex = 0
    key = key.upper()

    for symbol in ciphertext:  # Loop through each symbol in the message.
        num = LETTERS.find(symbol.upper())
        
        if num != -1:  # -1 means symbol.upper() was not found in LETTERS.
            num -= LETTERS.find(key[keyIndex])  # Subtract if decrypting.

            num %= len(LETTERS)  # Handle any wraparound.

            if symbol.isupper():
                decryption += LETTERS[num]
            elif symbol.islower():
                decryption += LETTERS[num].lower()

            keyIndex += 1  # Move to the next letter in the key.
            if keyIndex == len(key):
                keyIndex = 0
        else:
            # Append the symbol without encrypting/decrypting:
            decryption += symbol

    return decryption

def vigenereKeySolver(ciphertext: str, keylength: int):
    """
    return a list of the ten most likely keys
    """
    # Remove non characters in ciphertext
    ciphertext = re.compile('[^A-Z]').sub('',ciphertext.upper())

    # get subsequences of the ciphertext
    subsequences = getSubsequences(ciphertext, keylength)
    
    # For each subsequence get the Top IMC Dictionary
    mappings = []
    for subsequence in subsequences:
        max_score = calculateTopIMC(subsequence)
        top_5_values = dict(list(max_score.items())[:5])
        mappings.append(top_5_values)
    # print(mappings)

    permutations = []
    # doing cartesian product (combination of all k mappings with k letters) - Geeks for geeks
    for perm in itertools.product(*mappings):
        
        sum = 0
        for i, letter in enumerate(perm):
            sum += mappings[i][letter]
        
        permutations.append((perm, sum))
    
    #  Turn it into a dictionary
    perms = {}
    for i in permutations:
        perms[''.join(i[0])] = i[1]

    # Sort the dictionary and list top 10
    sorted_dict = dict(sorted(perms.items(), key=lambda item: item[1], reverse=True))
    top_10_keys = dict(list(sorted_dict.items())[:10])

    # Add top 10 strings to a list
    top_10 = []
    for i in top_10_keys:
        top_10.append(i)
    # print(top_10)

    return (top_10)

def hackVigenere(ciphertext: str):
    """
    return a string containing the key to the cipher
    """
    # Remove non characters in ciphertext
    ciphertext = re.compile('[^A-Z]').sub('',ciphertext.upper())

    keyLen = 0
    numkeys = 1
    while True:
        keyLenList = a7p234.keyLengthIC(ciphertext, numkeys)
        numkeys += 1
    
        if keyLenList[-1] < 10: # assignment says max length is 10
            keyLen = keyLenList[-1]
            break

    # Find the key of the ciphertext with key len that we found
    keys = vigenereKeySolver (ciphertext, keyLen)

    return (keys [0])


    # find first 5 possible key lengths
    # keyLens = []
    # numKeys = 1
    # count = 0
    # while True:
    #     keyLenList = a7p234.keyLengthIC(ciphertext, numKeys)
    #     numKeys += 1
        
    #     if keyLenList[-1] <= 10: # assignment says max length is 10
    #         keyLens.append(keyLenList[-1])
    #         count += 1
    #     if count == 3:
    #         break



    # keyLens = []
    # keyLenList = a7p234.keyLengthIC(ciphertext, 7)
        
    # for key in keyLenList:
    #     if key <= 10: # assignment says max length is 10
    #         keyLens.append(key)

    # key_dict = {}
    # # find top 10 keys for each key len
    # for keyLen in keyLens:
    #     # Find the key of the ciphertext with key len that we found
    #     keys = vigenereKeySolver(ciphertext, keyLen)
    #     key_dict[keyLen] = keys

    # ngrams = ngramsFreqsFromFile("wells.txt", 4)

    # score_dict = {}
    # for keys in key_dict.values():
    #     for key in keys:
    #         score = keyScore(key, ngrams, 4)
    #         score_dict[key] = score

    # max_val = max(score_dict, key=score_dict.get)
    
    # return(max_val)

def readFile(filename):
    # Read file and preprocess it

    file = open(filename, 'r')
    text = (file.read())
    file.close()

    # Remove non characters in ciphertext
    ciphertext = re.compile('[^A-Z]').sub('',text.upper())

    return(ciphertext)

def crackPassword():
    """
    hack password_protected.txt and write it to a new file
    """

    # read file
    ciphertext = readFile("password_protected.txt")

    # Get the key using the Function above
    key = hackVigenere(ciphertext)

    # Decipher the ciphertext
    message = decryptVigenere(ciphertext, key)
    Message = list(message)

    plain_text = []
    i = 0
    for char in ciphertext:
        if char not in LETTERS: # space / symbol
            plain_text.append(char)

        else:
            plain_text.append(Message[i]) # append deciphered letter
            i += 1 # next letter
    
    plain_text = "".join(plain_text)

    # Wite to file
    file = open("plaintext.txt", 'w')
    file.write(plain_text)
    file.close()

    return None

def test():
    # vigenereKeySolver Tests
    # print(getLetterFrequency("THAT")) # WORKS
    # print(getSubsequences("abcdefghi", 3)) # WORKS
    # print(decryptVigenere("qyxawuk", "hello")) # WORKS

    # vigenereKeySolver Tests
    ciphertext = "QPWKALVRXCQZIKGRBPFAEOMFLJMSDZVDHXCXJYEBIMTRQWNMEAIZRVKCVKVLXNEICFZPZCZZHKMLVZVZIZRRQWDKECHOSNYXXLSPMYKVQXJTDCIOMEEXDQVSRXLRLKZHOV"
    best_keys = vigenereKeySolver(ciphertext, 5)
    # print(best_keys)
    assert best_keys[0] == "EVERY"

    ciphertext = "DLGCGJKWQSWTOVWPYLKRBSEKRETSREKPMDSDPYLGSPUMLQSLDLGCTPYFJOQZEXGGMQRMRGEQXSRRXGCLYBH"
    best_keys = vigenereKeySolver(ciphertext, 3)
    # print(best_keys)
    assert best_keys[0] == "KEY"

    ciphertext = "VYCFNWEBZGHKPWMMCIOGQDOSTKFTEOBPBDZGUFUWXBJVDXGONCWRTAGYMBXVGUCRBRGURWTHGEMJZVYRQTGCWXF"
    best_keys = vigenereKeySolver(ciphertext, 6)
    # print(best_keys)
    assert best_keys[0] == "CRYPTO"

    ciphertext = "ANNMTVOAZPQYYPGYEZQPFEXMUFITOCZISINELOSGMMOAETIKDQGSYXTUTKIYUSKWYXATLCBLGGHGLLWZPEYXKFELIEUNMKJMLRMPSEYIPPOHAVMCRMUQVKTAZKKXVSOOVIEHKKNUMHMFYOAVVMITACZDIZQESKLHARKAVEUTBKXSNMHUNGTNKRKIETEJBJQGGZFQNUNFDEGUU"
    best_keys = vigenereKeySolver(ciphertext, 5)
    # print(best_keys)
    assert best_keys[0] == "MAGIC"

    ciphertext = "AQNRXXXSTNSKCEPUQRUETZWGLAQIOBFKUFMGWIFKSYARFJSFWSPVXHLEMVQXLSYFVDVMPFWTMVUSIVSQGVBMAREKEOWVACSGYXKDITYSKTEGLINCMMKLKDFRLLGNERZIUGITCWJVGHMPFEXLDIGGFXUEWJIHXXJVRHAWGFYMKMFVLBKAKEHHO"
    best_keys = vigenereKeySolver(ciphertext, 6)
    # print(best_keys)
    assert best_keys[0] == "SECRET"
    


    # hackVigenere Tests
    ciphertext = "ANNMTVOAZPQYYPGYEZQPFEXMUFITOCZISINELOSGMMOAETIKDQGSYXTUTKIYUSKWYXATLCBLGGHGLLWZPEYXKFELIEUNMKJMLRMPSEYIPPOHAVMCRMUQVKTAZKKXVSOOVIEHKKNUMHMFYOAVVMITACZDIZQESKLHARKAVEUTBKXSNMHUNGTNKRKIETEJBJQGGZFQNUNFDEGUU"
    key = hackVigenere(ciphertext)
    assert key == "MAGIC"

    ciphertext = "AQNRXXXSTNSKCEPUQRUETZWGLAQIOBFKUFMGWIFKSYARFJSFWSPVXHLEMVQXLSYFVDVMPFWTMVUSIVSQGVBMAREKEOWVACSGYXKDITYSKTEGLINCMMKLKDFRLLGNERZIUGITCWJVGHMPFEXLDIGGFXUEWJIHXXJVRHAWGFYMKMFVLBKAKEHHO"
    key = hackVigenere(ciphertext)
    assert key == "SECRET"

    ciphertext = "JDMJBQQHSEZNYAGVHDUJKCBQXPIOMUYPLEHQFWGVLRXWXZTKHWRUHKBUXPIGDCKFHBZKFZYWEQAVKCQXPVMMIKPMXRXEWFGCJDIIXQJKJKAGIPIOMRXWXZTKJUTZGEYOKFBLWPSSXLEJWVGQUOSUHLEPFFMFUNVVTBYJKZMUXARNBJBUSLZCJXETDFEIIJTGTPLVFMJDIIPFUJWTAMEHWKTPJOEXTGDSMCEUUOXZEJXWZVXLEQKYMGCAXFPYJYLKACIPEILKOLIKWMWXSLZFJWRVPRUHIMBQYKRUNPYJKTAPYOXDTQ"
    key = hackVigenere(ciphertext)
    assert key == "QWERTY"

    crackPassword()

if __name__ == '__main__' and not flags.interactive:
    test()