def validar_ADN(secuencia):
    secuencia=secuencia.upper()
    for i in secuencia:
        if i=="A" or i=="C" or i=="T" or i=="G":
            continue
        else:
            return("secuencia incorrecta")
    return("secuencia correcta")

      

def subsecuencia_ADN(secuencia,n):
    repetidos=[]
    norepetidos=[]
    if validar_ADN(secuencia):
        for i in range(len(secuencia)):
            if secuencia.count(secuencia[i:i+n])>1:
                if len(secuencia[i:i+n])==n:
                    repetidos.append(secuencia[i:i+n])
                    continue
                else:
                    repetidos.append('ninguna')
                    continue

            elif secuencia.count(secuencia[i:i+n])==1:
                if len(secuencia[i:i+n])==n:
                    norepetidos.append(secuencia[i:i+n])
                    continue
                else:
                    norepetidos.append('ninguna')
                    continue
        if norepetidos==[]:
            norepetidos.append('ninguna')
            return norepetidos
        elif norepetidos!=[]:
            for i in norepetidos:
                if norepetidos.count(i)>1:
                    norepetidos=[]
                    norepetidos.append('ninguna')
                    return norepetidos
                else:
                    return norepetidos
            
            


secuencia=input('ingresar secuencia:')
n=int(input('ingrese n:'))
print(subsecuencia_ADN(secuencia,n))