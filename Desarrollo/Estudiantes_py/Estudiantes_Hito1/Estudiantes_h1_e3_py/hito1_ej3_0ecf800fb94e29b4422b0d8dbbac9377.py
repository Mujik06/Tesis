#Aprobación de créditos
ingreso=int(input("introduzca los ingresos: "))
AN=int(input("ingrese año de nacimiento: "))
Edad=(2020-AN)
Hijos=int(input("ingrese la cantidad de hijos: "))
Pertenencia=int(input("ingrese años de pertenencia: "))
EC=str(input("ingrese estado civil (Soltero=S o Casado=C): "))
Residencia=str(input("ingrese residencia (Campo=R o Ciudad=U): "))
if(Pertenencia>=10 and Hijos>=2):
  print("APROBADO")
elif(EC=="C" and Hijos>3 and 45<=Edad<=55):
  print("APROBADO")
elif(ingreso>2500000 and EC=="S" and Residencia=="C"):
  print("APROBADO")
elif(ingreso>3500000 and Pertenencia>=5):
  print("APROBADO")
elif(Residencia=="R" and EC=="C" and Hijos<2):
  print("APROBADO")
else:
  print("RECHAZADO")