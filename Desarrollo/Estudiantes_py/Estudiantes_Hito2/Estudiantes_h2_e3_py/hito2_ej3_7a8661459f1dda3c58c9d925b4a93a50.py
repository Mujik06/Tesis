def secuencias(lista):

    palabras = []
    salida = []
    string = lista [0]
    n = lista[1]
    for i in range(0,len(string)-n+1):
        aux = ""
        for j in range(i,i+n):
            aux += string[j]
        palabras.append(aux)
    cont = 0
    for p in palabras:
        i = 0
        for j in palabras:
            if p == j:
                i += 1
        if i == 1:
            cont += 1
            salida.append(p.lower())
    if len(salida) == 0:
        salida.append('ninguna')
    return salida
if __name__ == "__main__":
    string = input('Texto: ')
    n = int(input('n: '))
    print(secuencias([string,n]))

