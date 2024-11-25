import cryptomath
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.'

def affine(key, message):
    """
    Decrypts a given cipher text and key using the Affine cipher

    Parameters:
        key: An integer representing the key. See the Chapter 14 to see how
             both keyA and keyB are encoded into a single integer key. 
        message: The string encrypted ciphertext
    """
    keyA = key // len(SYMBOLS)
    keyB = key % len(SYMBOLS)
    plaintext = ''
    modInverseOfKeyA = cryptomath.findModInverse(keyA, len(SYMBOLS))

    if modInverseOfKeyA is None:
        return '.'

    for symbol in message:
        if symbol in SYMBOLS:
            # Decrypt the symbol:
            symbolIndex = SYMBOLS.find(symbol)
            plaintext += SYMBOLS[(symbolIndex - keyB) * modInverseOfKeyA % len(SYMBOLS)]
        else:
            plaintext += symbol # Append the symbol without decrypting.
    return plaintext

def getKeyParts(key):
    keyA = key // len(SYMBOLS)
    keyB = key % len(SYMBOLS)
    return (keyA, keyB)