#Aprobación de créditos
Ingreso = int(input("ingrese su ingreso: "))
Anodenacimiento = int(input("ingrese su ano de nacimiento: "))
Numerodehijos = int(input("ingrese su numero de hijos: "))
Anosdepertenenciaalbanco = int(input("ingrese los anos de pertenencia al banco: "))
Estadocivil = str(input("ingrese su estado civil (S) soltero o (C) casado: "))
Siviveencampoocuidad = str(input("vive en campo o cuidad (U) urbano (R) rural: "))
Anoactual = 2021
Edad = Anoactual - Anodenacimiento
if Anosdepertenenciaalbanco > 10 and Numerodehijos >= 2:
 print ("APROBADO")
if str(Estadocivil) == "C" and Numerodehijos >= 3 and Edad > 45 and Edad < 55:
 print ("APROBADO")
if Ingreso > 2500000 and str(Estadocivil) == "S" and str(Siviveencampoocuidad) == "U":
 print ("APROBADO")
if Ingreso > 3500000 and Anosdepertenenciaalbanco > 5:
 print ("APROBADO")
if str(Siviveencampoocuidad) == "R" and str(Estadocivil) == "C" and Numerodehijos < 2:
 print ("APROBADO")
else:
 print ("RECHAZADO")