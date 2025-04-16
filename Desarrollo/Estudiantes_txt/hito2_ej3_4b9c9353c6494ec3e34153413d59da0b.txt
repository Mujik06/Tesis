def revisar_comienzo(cadena):
    while cadena[0] == "a":
        lista = list(cadena)
        lista.append(cadena[0])
        lista.pop(0)
        nueva_cadena = ""
        for letras in lista:
            nueva_cadena += letras
        cadena = nueva_cadena
    return cadena

def reordenar(cadena):
    lista = list(cadena)
    lista.append(cadena[0])
    lista.pop(0)
    nueva_cadena = ""
    for letras in lista:
        nueva_cadena += letras
    return nueva_cadena

def revisar_cadena(cadena,numero,lista):
    contador = 0
    inicio = cadena[:numero]
    if inicio not in lista:
        print(inicio, "no esta en", lista)
        cadena = cadena[numero:]
        lista.append(inicio)
        contador += 1
    else:
        cadena = revisar_comienzo(cadena)
        cadena = reordenar(cadena)
    return cadena,lista


cadena          = input("Ingrese Cadena: ").lower()
numero          = int(input("Ingrese numero: "))
largo_cadena    = len(cadena)
lista_cadenas   = []

if largo_cadena % numero == 0:
    while len(cadena) > 0:
        print(cadena)
        cadena = revisar_comienzo(cadena)
        cadena,lista_cadenas = revisar_cadena(cadena,numero,lista_cadenas)
    print(lista_cadenas)

else:
    print("ninguna")