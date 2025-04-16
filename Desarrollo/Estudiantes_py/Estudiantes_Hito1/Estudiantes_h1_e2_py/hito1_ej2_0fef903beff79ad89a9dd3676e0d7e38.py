#Contestador de celular
X=str(input("Ingrese el número de telefono: "))
Y = int(input("Ingrese hora de la llamada: "))
pr = X[0:3]
ul = X[5:8]
if 0 <= Y <= 7:
  print("CONTESTAR")
elif 7< Y <= 14:
  if ul == "909":
    print("CONTESTAR")
  else:
    print("NO CONTESTAR")
elif 17 <= Y <= 19:
  if pr == "877":
    print("NO CONTESTAR")
  else:
    print("CONTESTAR")
else: 
  print("NO CONTESTAR")      