secuencia=input("a:")
j=len(secuencia)
s=0
n=int(input("n:"))
lista_secuencias=[]
secuencial=secuencia[s:n]
lista_no=[]
lista_secuencias.append(secuencial)
while n<j:
    s=s+1
    n=n+1
    secuencia2=secuencia[s:n]
    if secuencia2 in lista_secuencias:
        lista_no.append(secuencia2)
    else:
        lista_secuencias.append(secuencia2)
for i in lista_no:
    if i in lista_secuencias:
        lista_secuencias.remove(i)
for i in range(len(lista_secuencias)):
    print(lista_secuencias[i])
    print("\n")
if len(lista_secuencias)==0:
    print("ninguna")