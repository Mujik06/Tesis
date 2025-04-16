string = input()
n = int(input())
lista = []
rep = []
for i in range(len(string) - n + 1):
    if string[i:i + n] not in lista and string[i:i + n] not in rep:
        lista.append(string[i:i + n])
    else:
        rep.append(string[i:i + n])
        if string[i:i+n] in lista:
          lista.pop(lista.index(string[i:i + n]))
if len(lista) > 0:
    [print(i) for i in lista]
else:
    print("ninguna")
