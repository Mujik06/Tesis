#Contestador de celular
numero= int(input("ingrese numero de telefono:"))
hora= int(input("ingrese hora de llamada:"))
numero_str= str(numero)
numero_lista= list(numero_str)
numero_tupla= tuple(numero_lista)
if 0<=hora<=7:
  print("CONTESTAR")
elif 7<hora<14:
  if numero_tupla[5]=="9" and numero_tupla[6]=="0" and numero_tupla[7]=="9":
    print ("CONTESTAR")
  else:
    print ("NO CONTESTAR")
elif 14<=hora<17:
  print ("NO CONTESTAR")
elif 17<=hora<=19:
  if numero_tupla[0]=="8" and numero_tupla[1]=="7" and numero_tupla[2]=="7":
    print ("NO CONTESTAR")
  else:
    print ("CONTESTAR")
else:
  print ("NO CONTESTAR")      