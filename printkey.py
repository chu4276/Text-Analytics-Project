prison_keyword = open('prison keyword.txt', 'r', encoding = 'utf-8').readlines()
appendixusedkeyword = open('appendixkey.txt', 'w')
def readfiles(file):
    keys=[]
    for word in file:
        text = word.rstrip('\n')
        text = text.rstrip(' ')
        keys.append(text)
        appendixusedkeyword.write(text + ', ')
    return keys

#print(readfiles(prison_keyword))
print(len(readfiles(prison_keyword)))


