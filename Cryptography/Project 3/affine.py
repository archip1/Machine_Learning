# Affine Cipher
# http://inventwithpython.com/hacking (BSD Licensed)

import sys, cryptomath, random

SYMBOLS = ''.join([chr(i) for i in range(ord(' '),ord('~')+1)])
#SYMBOLS = ''.join([chr(i) for i in range(ord('A'), ord('Z')+1)])

def affine(key, message, mode):
    keyA, keyB = getKeyParts(key)
    checkKeys(keyA, keyB, mode)
    translated = ''
    modInverseOfKeyA = cryptomath.findModInverse(keyA, len(SYMBOLS))
    for symbol in message:
        if symbol not in SYMBOLS:
            translated += symbol
        else:
            symIndex = SYMBOLS.find(symbol)
            if mode == 'encrypt':
                i = (symIndex * keyA + keyB) % len(SYMBOLS)
            else:    # mode == 'decrypt'
                i = (symIndex - keyB) * modInverseOfKeyA % len(SYMBOLS)
            translated += SYMBOLS[i]
    return translated

def getKeyParts(key):
    keyA = key // len(SYMBOLS)
    keyB = key % len(SYMBOLS)
    return (keyA, keyB)

def checkKeys(keyA, keyB, mode):
    if keyA == 1 and mode == 'encrypt':
        sys.exit('Key A is set to 1.')
    if keyB == 0 and mode == 'encrypt':
        sys.exit('Key B is set to 0.')
    if keyA < 0 or keyB < 0 or keyB > len(SYMBOLS) - 1:
        sys.exit('Key A and/or Key B out of range.')
    if cryptomath.gcd(keyA, len(SYMBOLS)) != 1:
        sys.exit('Key A is not relatively prime.')

def getRandomKey():
    while True:
        keyA = random.randint(2, len(SYMBOLS))
        keyB = random.randint(2, len(SYMBOLS))
        if cryptomath.gcd(keyA, len(SYMBOLS)) == 1:
            return keyA * len(SYMBOLS) + keyB

def main():
    myMessage = """"Can machines think?" -Alan Turing"""
    myKey = getRandomKey()

    print(SYMBOLS)
    print('Key:', myKey)
    print("Key A, Key B:", getKeyParts(myKey))
    print(' Plaintext:', myMessage)
    ciphertext = affine(myKey, myMessage, 'encrypt')
    print('Ciphertext:', ciphertext)
    plaintext = affine(myKey, ciphertext, 'decrypt')
    print(' Decrypted:', plaintext)

if __name__ == '__main__':
    main()
