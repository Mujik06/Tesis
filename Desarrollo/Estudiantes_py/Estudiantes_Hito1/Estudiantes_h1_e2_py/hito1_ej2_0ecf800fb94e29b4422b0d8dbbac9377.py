#Contestador de celular
telefono=int(input("ingrese telefono: "))
hora=int(input("ingrese hora: "))
caso1A=str(telefono)
caso1B=(caso1A[5:8])
caso1C=str(909)
caso2A=str(telefono)
caso2B=(caso2A[0:3])
caso2C=str(877)
if(0<=hora<=7):
  print("CONTESTAR")
if(7<hora<=14):
  if(caso1B==caso1C):
    print("CONTESTAR")
  else:
     print("NO CONTESTAR") 
if(17<=hora<=19):
  if(caso2B==caso2C):
    print("NO CONTESTAR")
  else:
    print("CONTESTAR")  
if(19<hora):
  print("NO CONTESTAR")