palabra=input("ingresa una palabra: ")
palabra=palabra.upper()
I=0
estado = True

while I<len(palabra):
    print(palabra[I])

    if palabra[I]!="A" and palabra[I]!="C" and palabra[I]!="T" and palabra[I]!="G":
        estado = False
        break
    else:
        pass
    I=I+1

if estado== False:
    print("Secuencia incorrecta")
if estado== True:
    print("Secuencia correcta")    