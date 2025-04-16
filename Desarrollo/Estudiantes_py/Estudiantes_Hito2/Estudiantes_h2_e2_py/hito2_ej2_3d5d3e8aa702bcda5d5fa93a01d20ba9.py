string=input("Ingrese una secuencia solo con las letras A,C,T,G")
string=string.upper()
string=list(string)
contador=0
for i in range(0,len(string)):
    if string[i]=="A" or string[i]=="C" or string[i]=="T" or string[i]=="G":
        contador+=1


if contador==len(string):
    print("secuencia correcta")
else:
    print("secuencia incorrecta")