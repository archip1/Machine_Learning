# Caesar Cipher
# http://inventwithpython.com/hacking (BSD Licensed)

import sys, util

NUM_SYM = 26

def caesar(key, message):
    translated = []

    for symbol in message:
        num = util.let2ind(symbol)
        if (num < 0) or (num >= NUM_SYM):
            translated.append(symbol)
        else:
            num = (num + key) % NUM_SYM
            translated.append(util.ind2let(num))
    
    return ''.join(translated)

def main():
    # message = util.getTextFromFile()
    message = "the end is near"
    message = message.upper()
    key = 7

    print(" Plaintext:", message)
    ctext = caesar(key, message)
    print("Ciphertext:", ctext)
    ptext = caesar(-key, ctext)
    print(" Decrypted:", ptext)
    exit(0)

    print("\nBrute-force decipherment:")
    for shift in range(NUM_SYM):
        guess = caesar(-shift, ctext)
        print("Key =", shift, guess)
    
if __name__ == '__main__':
    main()