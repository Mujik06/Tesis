num=(input("ingrese numero de telefono: "))
hora=int(input("ingrese hora de dia: "))
numero=[]
for i in num:
    numero.append(int(i))
print(numero)    

if hora>0 and hora<=7:
    print("contestar")
if hora>=19:
    print("no contestar")
if hora<=14:
    if numero[5]==9 and numero[6]==0 and numero[7]==9:
        print("contestar")
    else:
        print("no contestar")
if hora>=17 and hora<=19:
    if numero[0]==8 and numero[1]==7 and numero[2]==7:
        print("no contestar")
    else:
        print("contestar")    