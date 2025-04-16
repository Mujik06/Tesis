#Contestador de celular
a=input("ingrese su numero telefonico: ")
b=int(input("ingrese hora de la llamada: "))
c=int(a[5:])
d=int(a[:3])







if(7>b>=0):
    print("CONTESTAR")
if(14>b>=7):
    if(c==909):
      print("CONTESTAR")
    else :
      print("NO CONTESTAR")
if(17>b>=14):
    if(d==877):
      print("NO CONTESTAR")
    else :
        print("CONTESTAR")
if(24>b>17):
    print("NO CONTESTAR")