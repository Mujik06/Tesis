#Aprobación de créditos
ingreso = int(input("ingreso: "))
ano_nac = int(input("año nacimiento: "))
hijos = int(input("numero de hijos: "))
antiguedad = int(input("años de pertenencia al banco: "))
est_civil = input("estado civil mencionar S si es soltero o C si es casado: ")
sector = input("sector en el cual vive mencione U si es Urbano o R si es rural: ")

if antiguedad >= 10 and hijos >= 2:
    print("APROBADO")
elif est_civil == "C" and hijos > 3 and ano_nac <= 1976 and ano_nac >= 1966:
    print("APROBADO")
elif ingreso >= 2500000 and est_civil == "S" and sector == "U":
    print("APROBADO")
elif ingreso >= 3500000 and antiguedad > 5:
    print("APROBADO")
elif sector == "R" and est_civil == "C" and hijos < 2:
    print("APROBADO")
else:
    print("RECHAZADO")