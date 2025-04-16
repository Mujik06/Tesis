secuencia = str(input("Ingrese una secuencia de ADN: ")).upper()
s_lower = secuencia.upper()
ADN = ['A','C','G','T']

for x in secuencia:
    if x not in ADN:
        print("secuencia incorrecta")
        break
    if x in ADN:
        print("secuencia correcta")