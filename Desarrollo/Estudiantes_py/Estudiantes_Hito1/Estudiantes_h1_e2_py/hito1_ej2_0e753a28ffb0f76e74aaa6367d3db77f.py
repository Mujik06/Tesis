#Contestador de celular
#Contestador Automático
telefono=int(input("ingrese su número de teléfono :\n"))
hora=int(input("ingrese la hora de su llamada \n"))
if hora<=7:
  print ("Contestar")
if hora >7 and hora<14:
  if (telefono % 1000) == 909:
    print ("Contestar")
  else: 
    print ("No contestar")
if hora >=17 and hora<=19:
  if (telefono//100000)==877:
    print ("No contestar")
  else: 
    print ("Contestar")
if hora >19 and hora <= 23:
  print ("No Contestar")