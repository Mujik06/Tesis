string = input("Ingrese la secuencia: ")
numero= int(input("Ingrese el largo del sub string: "))

listasubstrings=[]
#substrings largo n
contador = 0
while contador < len(string) - numero +1:
    substring= string[contador: contador+numero]
    listasubstrings.append(substring)
    contador += 1

for elemento in listasubstrings:
    veces= listasubstrings.count(elemento)
    if veces == 1:
        print(elemento)
    else:
        print("ninguna")