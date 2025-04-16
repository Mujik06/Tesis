def buscartodas(a,b):
    lista_a=list(a)
    posiciones=[]
    for p in lista_a:
        if p==b:
            posicion=lista_a.index(p)
            lista_a[posicion]="-"
            posiciones.append(str(posicion))
    lugar="".join(posiciones)
    return lugar
secuencia=input('secuencia de adn:')
secuencia=secuencia.upper()
listaSec=[]
Secuencia=[]
i=0
while i<len(secuencia):
    j=0
    while (i+j)<len(secuencia):
        listaSec.append(secuencia[i:i+j+1])
        j=j+1
    i=i+1
for k in listaSec[:]:
    if len(k)==3:
        Secuencia.append(k)
for j in Secuencia[:]:
    if Secuencia.count(j)!=1:
        posiciones=list(buscartodas(Secuencia,j))
        a=0
        while a<len(posiciones):
            posiciones[a]=int(posiciones[a])
            a=a+1
        i=1
        while i<=len(posiciones):
            Secuencia[posiciones[i-1]]="-"
            i=i+1
l=0
while l<len(Secuencia):
    if Secuencia[l] is "-":
        Secuencia.remove('-')
        l=l
    else:
        l=l+1  
Secuencia="\n".join(Secuencia)
if len(Secuencia)>=1:
    print(Secuencia)
elif len(Secuencia)==0:
    print("ninguna")