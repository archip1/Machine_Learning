# this module contains some utility functions

import sys, os, re

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def let2ind(c):
    return ord(c) - ord('A')

def ind2let(i):
    return chr(i + ord('A'))

def getItemAtIndexZero(x):
    return x[0]

def getItemAtIndexOne(x):
    return x[1]

def sortByFreq(dict):
    itemList = list(dict.items())
    itemList.sort(key=getItemAtIndexOne, reverse=True)
    return itemList

def checkWord(regex):
    matches = []
    regex = regex.upper()
    wordFile = open('dictionary.txt')
    for line in wordFile:
        word = line[:-1]
        if re.match(regex, word):
            matches.append(word)
    return matches

def removeDupes(myString):
    newStr = ""
    for ch in myString:
        if ch not in newStr:
            newStr = newStr + ch
    return newStr

def removeMatches(myString,removeString):
    newStr = ""
    for ch in myString:
        if ch not in removeString:
            newStr = newStr + ch
    return newStr

def getKey(password):
    password = removeDupes(password)
    lastChar = password[-1]
    lastIdx = LETTERS.find(lastChar)
    afterString = removeMatches(LETTERS[lastIdx+1:],password)
    beforeString = removeMatches(LETTERS[:lastIdx],password)
    # print(password, afterString, beforeString)
    key = password + afterString + beforeString
    return key

def getTextFromFile():
    if len(sys.argv) < 2:
        print('Please provide the cipher filename as argument')
        sys.exit()
    filename = sys.argv[1]

    if not os.path.exists(filename):
        print('The file %s does not exist' % (filename))
        sys.exit()

    fileObj = open(filename)
    text = fileObj.read()[:-1]
    fileObj.close()
    return text