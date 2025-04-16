def sub(string,n):
    l=list(string)
    palabra=""
    nueva=[]
    nueva2=[]
    i=0
    j=n-1
    while i<len(l)-3:
        while j<len(l):
            nueva.append(l[i:j+1])
            j+=1
            i+=1
    for l in nueva:
        x=nueva.count(l)
        if x==1:
            nueva2.append(l)
    if nueva2==[]:
        print("ninguna")
    else:
        for j in nueva2:
            u="".join(j)
            print(u)

a=input("string")
b=int(input("n"))
sub(a,b)    