N=int(input("Ingrese numero de celular:"))
H=int(input("Ingrese la hora de llamada:"))
 
pn= N//100000
u3= N%1000
if(H>=00) and (H<=7):
 print ("CONTESTAR")
 
if(H>=8) and (H<=14) or (u3==909):
 print ("CONTESTAR")
else:
  print ("NO CONTESTAR")
 
if(H>=15) and (H<=16):
 print ("NO CONTESTAR")
if(H>=17) and (H<=19): 
 print ("CONTESTAR")
elif(H>=15) and (H<=19) or (pn==877):
 print ("NO CONTESTAR")

if (H>=20) and (H<=00):
    print ("NO CONTESTAR")


  
 