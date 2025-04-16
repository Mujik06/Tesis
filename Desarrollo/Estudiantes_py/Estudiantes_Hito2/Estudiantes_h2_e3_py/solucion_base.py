s=input("Inserte Secuencia de ADN: ").upper()
n=int(input("Inserte tamaño de la subsecuencia a buscar:" ))
l1=[]
for i in range(0,len(s)-n+1):
    s1=s[i:i+n]
    for j in range(0,len(s)-n+1):
        if j!=i:
            s2=s[j:j+n]
        else:
            continue
        if s1==s2:
            break
        else:
            if j==len(s)-n:
                l1.append(s1)
if len(l1)==0:
    print("ninguna")
else:
    for secuencia in l1:
        print(secuencia)






