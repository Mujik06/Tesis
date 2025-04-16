# Variables

ingresos = int(input("Sueldo liquido del postulante al credito: "))
año_nacimiento = int(input("Año nacimiento del postulante al credito: "))
n_hijos = int(input("Numero de hijos del postulante al credito: "))
años_en_el_banco = int(input("Años que lleva el postulante al credito afiliado al banco: "))
estado_civil = str(input("Estado civil del postulante al credito\nSoltero (S)\nCasado (C)\n: "))
estado_civil = str(estado_civil)
localidad = str(input("Residencia del postulante al credito\nCampo (U)\nCiudad (R)\n: "))
edad = (int(2021)-año_nacimiento)

# Procedimiento

if años_en_el_banco >= 10  and n_hijos >= 2:
    print("APROBADO")
elif estado_civil=="C" and n_hijos>3 and 45 < edad < 55:
    print("APROBADO")
elif ingresos > 2500000 and estado_civil == "s" and localidad == "r":
    print("APROBADO")
elif ingresos > 3500000 and años_en_elbanco > 5:
    print("APROBADO")
elif localidad == "R" and estado_civil == "C" and n_hijos < 2:
    print("APROBADO")
else:
    print("RECHAZADO")
    