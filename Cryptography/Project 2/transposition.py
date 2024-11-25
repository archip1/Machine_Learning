# Transposition Cipher Encryption/Decryption/Test
# http://inventwithpython.com/hacking (BSD Licensed)

import random, sys, math, util

def encryptMessage(key, message):
    ciphertext = [''] * key

    for col in range(key):
        for pointer in range(col, len(message), key):
            ciphertext[col] += message[pointer]

    return ''.join(ciphertext)

def decryptMessage(key, message):
    numOfColumns = math.ceil(len(message) / key)
    numOfRows = key
    numOfShadedBoxes = (numOfColumns * numOfRows) - len(message)

    plaintext = [''] * numOfColumns

    col = 0
    row = 0

    for symbol in message:
        plaintext[col] += symbol
        col += 1 

        if (col == numOfColumns) or (col == numOfColumns - 1 and row >= numOfRows - numOfShadedBoxes):
            col = 0
            row += 1

    return ''.join(plaintext)

def transpTest():
    random.seed(42)

    for i in range(20):
        message = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' * random.randint(4, 40)
        message = list(message)
        random.shuffle(message)
        message = ''.join(message)
        print('Test #%s: "%s..."' % (i+1, message[:50]))

        for key in range(1, len(message)):
            encrypted = encryptMessage(key, message)
            decrypted = decryptMessage(key, encrypted)

            if message != decrypted:
                print('Mismatch with key %s and message %s.' % (key, message))
                print(decrypted)
                sys.exit()

    print('Transposition cipher test passed.')

def main():
    myMessage = util.getTextFromFile()
    myKey = 8

    print(myMessage)
    ciphertext = encryptMessage(myKey, myMessage)
    print(ciphertext)
    plaintext = decryptMessage(myKey, ciphertext)
    print(plaintext)
    transpTest()

if __name__ == '__main__':
    main()