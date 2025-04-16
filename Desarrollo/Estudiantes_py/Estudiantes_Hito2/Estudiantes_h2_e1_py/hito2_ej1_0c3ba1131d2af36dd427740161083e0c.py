secuencia=input()
i=0
j=0

while(i<len(secuencia)):
    if(secuencia[i]=="A") or (secuencia[i]=="a"):
        i=i+1
        j=j+1
        
    elif(secuencia[i]=="C") or (secuencia[i]=="c"):
        i=i+1
        j=j+1
    elif(secuencia[i]=="T") or (secuencia[i]=="t"):
        i=i+1
        j=j+1
    elif(secuencia[i]=="G") or (secuencia[i]=="g"):
        i=i+1
        j=j+1
    else:
        i=i+1
if(j==len(secuencia)):
    print("secuencia correcta")
else:
    print("secuencia incorrecta")     