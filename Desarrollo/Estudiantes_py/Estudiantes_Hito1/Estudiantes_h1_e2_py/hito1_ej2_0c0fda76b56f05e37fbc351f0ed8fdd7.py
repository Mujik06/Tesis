#Contestador de celular
a=int(input("ingrese le numero telefonico 8 dij.: "))
b=int(input("ingrese la hora: "))
if (b>=0 and b<=7):
  print("CONTESTAR")
if(b>7 and b<=14):
   if(a%1000==909):
      print ("CONTESTAR")
   else:
      print ("NO CONTESTAR")
if(b>=17 and b<=19):
    if (a//100000==877):
      print("NO CONTESTAR")
    else:
      print("CONTESTAR")
if(b>19):
    print ("NO CONTESTAR")    