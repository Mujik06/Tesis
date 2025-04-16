bases=["A","C","T","G"," "]
secuencia=input()
def genoma(secuencia):
    valor=True
    for base in secuencia:
        if base not in bases:
            valor=False
    return valor
if genoma(secuencia)==True:
    print("secuencia correcta")
else:
    print("secuencia incorrecta")
