#Contestador de celular
n=int(input())
h=int(input())
t=n-(int(n/1000)*1000)
c=int(n/100000)
if 0<=h<=7:
    print("CONTESTAR")
elif 14<h and t!=909:
    print("NO CONTESTAR")
elif t==909:
    print("CONTESTAR")
elif 17<=h<=19 and c!=877:
    print("CONTESTAR")
elif h>19:
    print("NO CONTESTAR") 

