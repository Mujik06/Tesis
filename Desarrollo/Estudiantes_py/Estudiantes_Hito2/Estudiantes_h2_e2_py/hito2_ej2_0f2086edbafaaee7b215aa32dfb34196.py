genoma=input("secuencia genoma:")
genoma=genoma.upper()
a=genoma.find("A")
b=genoma.find("C")
c=genoma.find("T")
d=genoma.find("G")
if a!=-1 and b!=-1 and c!=-1 and d!=-1 and b==-1:
    print("secuencia correcta")
else:
    print("secuencia incorrecta")