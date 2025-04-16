secuencia=input("ingrese secuencia de ADN: ")
secuencia.upper()
t=" "
for t in secuencia:
    if t!="A" or t!="C" or t!="T" or t!="G":
        print("secuencia incorrecta")
    else:
        print("secuencia correcta")