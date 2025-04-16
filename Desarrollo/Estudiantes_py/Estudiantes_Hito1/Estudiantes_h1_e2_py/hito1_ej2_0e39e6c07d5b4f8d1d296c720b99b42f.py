#Contestador de celular
numero=int(input("ingrese numero: "))
hora=int(input("ingrese hora: "))
if 0<=hora<=7:
  print("CONTESTAR")
elif hora<14 and numero%1000==909: 
  print("CONTESTAR")
elif hora<14: 
  print("NO CONTESTAR")
numero1=format(numero/100000)
if 17<=hora<=19 and numero1==877:
  print("CONTESTAR")
elif 17<=hora<=19:
  print("NO CONTESTAR")
elif 19<hora:
  print("NO CONTESTAR")
  


