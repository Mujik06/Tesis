s=input("Ingrese su secuencia: ")
n=int(input("ingrese el largo de su subsecuencia: "))
for i in range (len(s)-n+1):
    x=0
    r=s
    for j in range (len(s)-n+1):
        if s[i:n+i]==r[:n]:
            x+=1
        r=r[1:]
    if x==1:
        print(s[i:n+i])
    else:
        print("ninguna")