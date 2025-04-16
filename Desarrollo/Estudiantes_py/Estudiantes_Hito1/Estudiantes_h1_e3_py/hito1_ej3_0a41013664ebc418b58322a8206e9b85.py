#Aprobación de créditos
ingreso= int(input("ingreso:"))
nacimiento= int(input("nacimiento:"))
hijos= int(input("hijos:"))
antiguedad= int(input("antiguedad:"))
estado_civil= str(input("estado civil:"))
campo_ciudad= str(input("campo o ciudad:"))
edad= 2020- nacimiento
if antiguedad>10 and hijos>=2:
  print ("APROBADO")
elif estado_civil=="C" and hijos>3 and 45<= edad<=55:
  print ("APROBADO")
elif ingreso>2500000 and estado_civil=="S" and campo_ciudad=="U":
  print ("APROBADO")
elif ingreso>3500000 and antiguedad>5:
  print ("APROBADO")
elif campo_ciudad=="R" and estado_civil=="C" and hijos<2:
  print ("APROBADO")
else:
  print ("RECHAZADO")      