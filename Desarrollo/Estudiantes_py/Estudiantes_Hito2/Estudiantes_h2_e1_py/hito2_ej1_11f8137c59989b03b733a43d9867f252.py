string1 = "ACCTGGTTCTGTAGTCAGGATTACTA"
string2 = "TGACGTTCAGTAGTCGATT"

lista1 = list(string1)
lista2 = list(string2)
alineado1 = []
cierto=0
alineado2 = []
while cierto==0:
    i = 0
    j = 0
    if lista1[i] != lista2[j]:
        alineado1.append(lista1[i])
        alineado2.append("_")
        i = i+1
    elif lista1[i]==lista2[j]:
        alineado1.append(lista1[i])
        alineado2.append(lista2[j])
        i = i+1
        j = j+1
    if j>len(lista2) or i> len(lista1):
        cierto = 1

print(alineado1)
print(alineado2)