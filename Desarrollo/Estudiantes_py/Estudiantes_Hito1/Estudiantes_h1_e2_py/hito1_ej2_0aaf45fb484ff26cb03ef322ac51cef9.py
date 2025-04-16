#Contestador de celular
numero = eval(input("Ingrese numero telefonico: "))
hora = eval(input("Ingrese hora de la llamada: "))

if hora >= 0 and hora <= 7:
 print("Resultado:Contestar")
elif hora <= 14:
 digitos = numero % 1000
 if digitos == 909:
  print("RESULTADO:CONTESTAR")
 else:
  print("RESULTADO:NO CONTESTAR")
elif hora >= 17 and hora <=19:
 digitos = numero // 100000
 if digitos == 877:
  print("RESULTADO:NO CONTESTAR")
 else:
  print("RESULTADO:CONTESTAR")
elif hora >= 19:
  print("RESULTADO:NO CONTESTAR")
else:
  print("RESULTADO:NO CONTESTAR")
