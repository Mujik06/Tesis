def alineamiento(string1,string2):
    palabra = ""
    tam1 = 0
    dif = 0
    if len(string1) > len(string2):
        tam1 = len(string2)
        dif = len(string1)-tam1
        
    else:
        tam1 = len(string1)
        dif = len(string2)-tam1
    i = 0
    k = 0
    while i < tam1+dif:
        for j in range(i,tam1+dif):
            if string2[k] == string1[j]:
                palabra += string2[k]
                k = k+1
                i = j
                break
            else:
                palabra += "_"
        i += 1
    for i in range(k,len(string2)):
        palabra += string2[i]

    return palabra
if __name__ == "__main__":
    string1 = input('String 1: ')
    string2 = input('String 2: ')
    print(alineamiento(string1,string2))
