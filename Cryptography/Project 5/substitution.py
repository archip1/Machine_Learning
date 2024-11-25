# Simple Substitution Cipher
# http://inventwithpython.com/hacking (BSD Licensed)

import sys, random

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def substitution(key, message, mode):
    translated = ''
    charsA = LETTERS
    charsB = key
    if mode == 'decrypt':
        charsA, charsB = charsB, charsA

    for symbol in message:
        if symbol.upper() in charsA:
            symIndex = charsA.find(symbol.upper())
            if symbol.isupper():
                translated += charsB[symIndex].upper()
            else:
                translated += charsB[symIndex].lower()
        else:
            translated += symbol

    return translated

def checkValidKey(key):
    if sorted(list(key)) != sorted(list(LETTERS)):
        sys.exit('There is an error in the key or symbol set.')

def getRandomKey():
    key = list(LETTERS)
    random.shuffle(key)
    return ''.join(key)

def main():
    myMessage = "There is no substitute for hard work. -Thomas A. Edison"
    myKey = 'LFWOAYUISVKMNXPBDCRJTQEGHZ'
    myKey = getRandomKey()

    print('Using key %s' % (myKey))
    checkValidKey(myKey)

    print(myMessage)
    ciphertext = substitution(myKey, myMessage, 'encrypt')
    print(ciphertext)
    plaintext = substitution(myKey, ciphertext, 'decrypt')
    print(plaintext)

if __name__ == '__main__':
    main()