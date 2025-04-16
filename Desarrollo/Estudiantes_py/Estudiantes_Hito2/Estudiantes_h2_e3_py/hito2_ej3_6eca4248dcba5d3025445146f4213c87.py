def subSecuencia(a,n):
    import random
    from random import shuffle
    
    i=0
    lista1=[]
    while i<len(a):
        lista1.append(a[i])
        i=i+1

    def potencia(c):
        if len(c) == 0:
            return [[]]
        r = potencia(c[:-1])
        return r + [s + [c[-1]] for s in r]

    def combinaciones(c, n):
        return [s for s in potencia(c) if len(s) == n]

    z=combinaciones(lista1,3)
    lista2=[]
    j=0
    while j<len(z):
        if z.count(z[j])==1 and z[j][0]!=z[j][1] and z[j][1]!=z[j][2] and z[j][0]!=z[j][2] and z[j]:
            lista2.append(z[j])
        j=j+1
    lista3=[]
    k=0
    while k<len(lista2)-1:
        if lista2[k][::-1]==lista2[k+1]:
            lista3.append(lista2[k])
        k=k+1
    lista4=[]
    l=0
    while l<len(lista3):
        lista4.append("".join(lista3[l]))
        l=l+1
    m=0
    lista5=[]
    while m<len(lista4):
        lista5.append(lista4[m].lower())
        m=m+1
    if len(lista5)!=0:
        return lista5
    return ["ninguna"]
