# Vigenere Cipher Dictionary Hacker
# http://inventwithpython.com/hacking (BSD Licensed)

import sys, os, util, detectEnglish, vigenere, substitution

def hack(cipherType, ciphertext):
    fo = open('dictionary.txt')
    words = fo.readlines()
    fo.close()

    for word in words:
        word = word.strip() 
        if cipherType == 'vigenere':
            decrypted = vigenere.vigenere(word, ciphertext, 'decrypt')
            maybe = detectEnglish.isEnglish(decrypted, wordPercentage=40)
        elif cipherType == 'substitution':
            key = util.getKey(word)
            decrypted = substitution.substitution(key, ciphertext, 'decrypt')
            maybe = detectEnglish.isEnglish(decrypted, wordPercentage=70)
        else:
            print("Unknown cipher type: %s" % cipherType)
            sys.exit()

        if maybe:
            print('\nPossible encryption break:')
            print('Key ' + str(word) + ': ' + decrypted[:100])
            print('\nEnter D for done, or press Enter')
            response = input('> ')

            if response.upper().startswith('D'):
                return decrypted
    return None

def main():
    cipherText = util.getTextFromFile()
    if len(sys.argv) > 2:
        cipherType = sys.argv[2]
    else:
        cipherType = 'vigenere'

    hackedMessage = hack(cipherType, cipherText)

    if hackedMessage == None:
        print('Failed to hack encryption.')
    else:
        print(hackedMessage)

if __name__ == '__main__':
    main()