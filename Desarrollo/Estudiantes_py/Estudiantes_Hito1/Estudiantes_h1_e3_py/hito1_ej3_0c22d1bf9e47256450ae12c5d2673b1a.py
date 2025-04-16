#Aprobación de créditos

Ingreso = int(input("Ingrese su monto en pesos:"))
AnoNacimiento = int(input("Ingrese su año de nacimiento:"))
NumeroHijos = int(input("Ingrese cantidad de hijos:"))
AnosEnBanco = int(input("Ingrese años de pertenencia al banco]:"))
Estado = str(input("Ingrese si es Soltero (S) o casado (C):"))
CampoOCiudad = str(input("Ingrese si vive en campo (R) o ciudad (U):"))

AnoActual = 2022
Edad = AnoActual - AnoNacimiento

if AnosEnBanco > 10 and NumeroHijos >= 2:
  print("APROBADO")

if str(Estado) == "C" and NumeroHijos > 3 and Edad > 45 and Edad < 55:
  print("APROBADO")

if Ingreso > 2500000 and str(Estado) == "S" and str(CampoOCiudad) == "U":
  print("APROBADO")

if Ingreso > 3500000 and AnosEnBanco > 5:
  print("APROBADO")

if str(CampoOCiudad) == "R" and str(Estado) == "C" and NumeroHijos < 2:
  print("APROBADO")