#Aprobación de créditos
from datetime import date

fecha = date.today()
ingreso = int(input("Ingreso (en pesos y sin puntos) "))
Nacimiento = int(input("Ingrese Año de nacimiento "))
Hijos = int(input("Ingrese Número de hijos "))
pertenencia = int(input("Ingrese Años de pertenencia al banco "))
estadocivil =  str(input("Ingrese Estado civil S (si es soltero), C (si es casado) "))
vive = str(input("Ingrese si vive en campo o cuidad U (urbano), R (rural)  "))
edad = fecha.year - Nacimiento

if pertenencia >= 10 and Hijos >= 2:
    print("APROBADO")
elif estadocivil == "C" and   Hijos > 3 and edad > 45 and edad < 55:
    print("APROBADO")
elif ingreso > 2500000 and estadocivil == "S" and vive == "U":
    print("APROBADO")
elif ingreso > 3500000 and pertenencia > 5:
    print("APROBBADO")
elif vive == "R" and estadocivil == "C" and Hijos < 2:
    print("APROBADO")
else:
    print("RECHAZADO")