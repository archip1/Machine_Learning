# Vigenere Cipher (Polyalphabetic Substitution Cipher)
# http://inventwithpython.com/hacking (BSD Licensed)

import os, util

NUM_SYM = 26

def vigenere(key, message, mode):
    translated = [] 
    keyIndex = 0
    key = key.upper()

    for symbol in message: 
        num = util.let2ind(symbol.upper())
        if (num < 0) or (num >= NUM_SYM):
            translated.append(symbol)
        else:
            if mode == 'encrypt':
                num = num + util.let2ind(key[keyIndex])
            elif mode == 'decrypt':
                num = num - util.let2ind(key[keyIndex])

            nextLetter = util.ind2let(num % NUM_SYM)
            if symbol.isupper():
                translated.append(nextLetter)
            elif symbol.islower():
                translated.append(nextLetter.lower())

            keyIndex = (keyIndex + 1) % len(key)

    return ''.join(translated)

def main():
    #myMessage = util.getTextFromFile()
    myMessage = "Focus on learning and good grades will follow."
    myKey = 'ASIMOV'

    print(myMessage)
    ciphertext = vigenere(myKey, myMessage, 'encrypt')
    print(ciphertext)
    plaintext = vigenere(myKey, ciphertext, 'decrypt')
    print(plaintext)

if __name__ == '__main__':
    main()