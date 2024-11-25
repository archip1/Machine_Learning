languages = ["English", "French", "Italian", "Spanish", "German", "Dutch", "Polish", "Russian", "Bulgarian", "Greek"]
languages.sort()

sumlen = 0
for i in languages:
     sumlen += len(i)
width = (sumlen + 4 * 10 +4)
print("-"* width)

line = "|           | "
for i in languages:
    if i != "[" or i != "]":
            line+= " "
            line+= str(i)
            line+= " |"
print(line)
print("-"* width)
list_of_numbers = [[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]

for i in range(len(list_of_numbers)):
    line = "| "
    line += languages[i]
    len = len(languages[i])
    add_space = 11 - len
    line += " |"
    for j in list_of_numbers[i]:
        if j != "[" or j!= "]":
            line+= " "
            line+= str(j)
            line+= " |"
    print (line)
print("-"* width)
