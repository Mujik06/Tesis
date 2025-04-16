n = int(input("ingrese numero telefonico: "))
hora = int(input("ingrese hora de llamada: "))

if hora >= 0 and hora <= 7:
 print("CONTESTAR")
elif hora < 14:
 if 10000909 <= n <= 99999909:
  print("CONTESTAR")
 else:     
  print("NO CONTESTAR")

elif hora >= 17 and hora <= 19:
 if n >= 87700000 and n <= 87799999:
  print("NO CONTESTAR")
 print("CONTESTAR")
else:
 print("NO CONTESTAR")
