ingreso = int(input("ingresos: "))
edad= int(input("edad: "))
hijos = int(input("numero de hijos: "))
pertenencia = int(input("años de pertenencia al banco: "))
estado_civil = input("S= soltero, o C= casado.: ")
vive = input("U= urbano, R= rural.: ")

if pertenencia > 10 and hijos >= 2:
 print("APROBADO")
elif estado_civil == "C" and hijos > 3 and edad >= 45 and edad <= 55:
 print("APROBADO")
elif ingreso > 2500000 and estado_civil == "S" and vive == "U":
 print("APROBADO")
elif ingreso > 3500000 and pertenencia > 5:
 print("APROBADO")
elif vive == "R" and estado_civil == "C" and hijos < 2:
 print("APROBADO")
else:
 print("RECHAZADO")