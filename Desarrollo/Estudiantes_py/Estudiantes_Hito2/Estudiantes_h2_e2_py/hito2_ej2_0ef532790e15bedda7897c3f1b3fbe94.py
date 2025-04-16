Secuencia=input("Ingrese secuencia:")
Secuencia=str.upper(Secuencia)
for letra in Secuencia:
    if letra!="A" and letra!="G" and letra!="C" and letra!="T":
        print("Secuencia incorrecta")
    else:
        print("Secuencia correcta")
