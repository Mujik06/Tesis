secuencia1 = input("Inserte Secuencia 1: ")
secuencia2 = input("Inserte secuencia 2: ")


secuencia1ComoLista = list(secuencia1)
secuencia2ComoLista = list(secuencia2)
if len(secuencia1ComoLista) == len(secuencia2ComoLista):
    letra = 0
    while letra < len(secuencia1ComoLista):
        if secuencia1ComoLista[letra] == secuencia2ComoLista[letra]:
            letra += 1
        else:
            secuencia1ComoLista.insert(letra, "*")
            letra += 1

if len(secuencia1ComoLista) != len(secuencia2ComoLista):
    if len(secuencia1ComoLista) < len(secuencia2ComoLista):
        letra = 0
        while letra < len(secuencia2ComoLista):
            if secuencia2ComoLista[letra] == secuencia2ComoLista[letra]:
                letra += 1
            else:
                secuencia1ComoLista.insert(letra, "_")
                letra += 1


    else:
        letra = 0
        while letra < len(secuencia1ComoLista):
            if secuencia1ComoLista[letra] == secuencia2ComoLista[letra]:
                letra += 1
            else:
                secuencia2ComoLista.insert(letra, "_")
                letra += 1
print("".join(secuencia1ComoLista))
print("".join(secuencia2ComoLista))