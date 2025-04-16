n=input("Ingrese Numero Telefónico:")
m=int(input("Ingrese hora de la llamada:"))
if (0<=m<=7):
    print("Contestar")

else:
    if(7<m<=14):
        o = n[5]
        p = n[6]
        q = n[7]
        if (o=="9" and p=="0" and q=="9"):
            print("Contestar")
        else:
                print("No Contestar")
    else:

             if (14<m<=17):
                 print("No Contestar")
if(17<m<=19):
    r = n[0]
    s = n[1]
    t = n[2]
    if(r=="8" and s=="7" and t=="7"):
        print("No Contestar")
    else:
        print("Contestar")

elif (19<m<=23):
    print("No Contestar")






