sec1 = input()
sec2 = input()
lista = []
r = 0
for i in range(len(sec2)):
    while (r + i) < len(sec1):
        if sec2[i] != sec1[i + r]:
            lista.append("_")
            r += 1
        else:
            break
    lista.append(sec2[i])

print("".join(lista))