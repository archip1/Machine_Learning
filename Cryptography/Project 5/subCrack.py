# this module contains helper functions for cracking substitution ciphers

import re, util

# global variables
alphabet = "abcdefghijklmnopqrstuvwxyz"

def letterFrequency(text):
    text = text.lower()
    nonletters = util.removeMatches(text,alphabet)
    nonletters = util.removeDupes(nonletters)
    text = util.removeMatches(text,nonletters)
    lcount = {}
    total = len(text)
    for ch in text:
        lcount[ch] = lcount.get(ch,0) + 1
    for ch in lcount:
        lcount[ch] = lcount[ch] / total
    return lcount

def maybeAdd(ch, neighbors):
    if ch in alphabet:
        neighbors[ch] = neighbors.setdefault(ch,0) + 1   

def neighborCount(text):
    nbDict = {}
    text = text.lower()
    for i in range(len(text)-1):
        ch = text[i]
        nextch = text[i+1]
        dict = nbDict.setdefault(ch,{})
        maybeAdd(nextch,dict)
        dict = nbDict.setdefault(nextch,{})
        maybeAdd(ch,dict)
    return nbDict

def checkWord(unused,pattern):
    resList = []
    wordFile = open('wordlist.txt')
    rePat = '['+unused+']'
    regex = re.sub('[a-z]',rePat,pattern) + '$'  
    regex = regex.lower()
    print('matching ', regex)
    for line in wordFile:
        if re.match(regex,line[:-1]):
            resList.append(line[:-1])
    return resList

def findLetters(unused,pattern):
    resList = []
    wordFile = open('wordlist.txt')
    ctLetters = re.findall('[a-z]',pattern)   
    print(ctLetters)
    rePat = '(['+unused+'])'  
    regex = re.sub('[a-z]',rePat,pattern) + '$'
    regex = regex.lower()
    for line in wordFile:
        myMatch = re.match(regex,line[:-1])
        if myMatch:                            
            matchingLetters = myMatch.groups()
            matchList = []
            for l in matchingLetters:
                matchList.append(l.upper())    
            resList.append(line[:-1])
            resList.append(list(zip(ctLetters,matchList)))  
    return resList

def showCounts(dict):
    print("   ",end="")
    for i in alphabet:
        print(i,end= "  ")
    print()
    for i in alphabet:
        print(i,end= " ")
        if i not in dict.keys():
            dict[i] = {}
        zeros = 0
        for j in alphabet:
            if j not in dict[i]:
                zeros = zeros + 1
            value = dict[i].setdefault(j,0)   
            print("%2d " % (value), end="")
        print('\t', zeros)
        #print()

def getFreq(t):
    return t[1]

def sortByFreq(dict):
    itemList = list(dict.items())
    itemList.sort(key=getFreq, reverse=True)
    return itemList

def getKeyLen(t):
    return len(t[0])

def sortByKeyLen(dict):
    itemList = list(dict.items())
    itemList.sort(key=getKeyLen)
    return itemList

def showLetterFrequency(text):
    letCount = letterFrequency(text)
    letList = sortByFreq(letCount)
    for entry in letList:
        print("%s %5.3f" % entry, end=' ')
    print()

def countWords(text):
    text = text.lower()
    wList = text.split()
    wCount = {}
    for w in wList:
        if w in wCount:
            wCount[w] = wCount[w] + 1
        else:
            wCount[w] = 1
    return wCount
        
def guess(unused, cLet, pLet, cipherText):
    for i in range(len(cLet)):
        cipherText = cipherText.replace(cLet[i],pLet[i])
        unused = unused.replace(pLet[i].lower(), '')
    return (unused, cipherText)
    
def main():
    plainFile = open("../p/mobydick.txt")
    plainText = plainFile.read()
    #showLetterFrequency(plainText)

    cipher = util.getTextFromFile()
    showLetterFrequency(cipher)
    #showCounts(neighborCount(cipher))
  
    unused = alphabet
    (unused,cipher) = guess(unused,'kuliw', 'THERC', cipher)
    wordDict = countWords(cipher)
    print(sortByFreq(wordDict))
    #print(sortByKeyLen(wordDict))
    print(cipher)
    print(findLetters(unused,'EnveEbwE'))

if __name__ == "__main__":
    main()

