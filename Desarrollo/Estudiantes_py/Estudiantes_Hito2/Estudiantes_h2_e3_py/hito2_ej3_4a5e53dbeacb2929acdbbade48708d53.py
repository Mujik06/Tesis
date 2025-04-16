a=input("")
n=int(input(""))
lista=[]
listam=[]
for b in range(n,len(a)+1): 
 g=a[b-n:b] 
 if(g in lista):
     lista.remove(g)
     listam.append(g)
 if (g not in listam):
  lista.append(g)
if(lista!=[]):
 for a in lista:
    print(a)
else:
    print("ninguna")
         