# Affine and Transposition Cipher Hacker
# http://inventwithpython.com/hacking (BSD Licensed)

import os, sys, detectEnglish, cryptomath, caesar, affine, transposition
from itertools import permutations

SILENT_MODE = False

def hack(cipherType, ciphertext, count, dictionary):
    if cipherType == 'caesar':
        keyRange = range(len(dictionary))
    elif cipherType == 'transposition':
        keyRange = []
        numpossiblecolumns = [*range(2,11)]
        for i in numpossiblecolumns:
            numcolumns = [*range(1, i+1)]
            perms = (permutations(numcolumns, i))
            for j in perms:
                keyRange.append(list(j))
    elif cipherType == 'affine':
        keyRange = range(len(affine.SYMBOLS) ** 2)
    else:
        print("Unknown cipher type: %s" % cipherType)
        sys.exit()

    print('Hacking...')
    print('(Press Ctrl-C or Ctrl-D to quit at any time.)')

    for key in keyRange:
        if cipherType == 'caesar':
            decrypted = caesar.caesar(dictionary[key], ciphertext)
        elif cipherType == 'transposition':
            decrypted = transposition.decryptMessage(key, ciphertext)
        elif cipherType == 'affine':
            keyA = affine.getKeyParts(key)[0]
            if cryptomath.gcd(keyA, len(affine.SYMBOLS)) != 1:
                continue
            decrypted = affine.affine(key, ciphertext)

        if not SILENT_MODE:
            print('Tried Key %s... (%s)' % (key, decrypted[:40]))

        if detectEnglish.isEnglish(decrypted):
            print("\n", count, ": ", cipherType)
            # print('\nPossible encryption hack:')
            print('Key %s: %s' % (key, decrypted))
            print('\nEnter D for done, or press Enter')
            response = input('> ')

            if response.strip().upper().startswith('D'):
                return decrypted    

    #         print(decrypted[:200])
    return None

def main():
    cipherList = [] # create an empty list to store all present words
    dictionaryWords = []
    filename = "ciphers_version2.txt"

    file = open(filename, 'r')
    for line in file:
        line = line[:-1]
        cipherList.append(line)
    file.close()

    filename = "dictionary.txt"
    file = open(filename, 'r')
    for line in file:
        line = line[:-1]
        dictionaryWords.append(line)
    file.close()

    cipherTypeList = ['transposition', 'affine', 'transposition', 'transposition', 'caesar', 'affine', 'caesar', 'transposition', 'caesar', 'affine', 'caesar', 'affine']

    # ciphertext = util.getTextFromFile()
    # print(hack('affine', cipherList[11], 0, dictionaryWords))
    print(hack('affine', "92w?ew2pmwj?SSme2wYKB2?bBmwE0wMBBwvKYjmHew?vw2pmwem2z", 0, dictionaryWords))
    # count = 0
    # for i in range(len(cipherList)):
    #     count+=1
    #     hack(cipherTypeList[i], cipherList[i], count, dictionaryWords)

            # if hackedMessage == None:
            #     print('Failed to hack encryption.')
            # else:
            #     print(hackedMessage)

            # if hackedMessage != None:
            #     print(hackedMessage)

    # cipherType = 'caesar'
    # if len(sys.argv) > 2:
    #     cipherType = sys.argv[2]

if __name__ == '__main__':
    main()
