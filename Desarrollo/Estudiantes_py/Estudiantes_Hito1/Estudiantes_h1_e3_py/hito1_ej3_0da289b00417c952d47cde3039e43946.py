#Aprobación de créditos
print("Welcome to THE Bank")

ingreso = int(input("Ingreso mensual en pesos: "))
año_nacimiento = int(input("Año de nacimiento: "))
n_hijos = int(input("Número de hijos: "))
años_de_pertenencia = int(input("Años como cliente del banco uwu: "))
estado_civil = str(input("Estado civil (S para soltero, C para casado): "))
residencia = str(input("Lugar de residencia (R for rural, U for urbano): "))
año = 2020
edad = (int(año) - int(año_nacimiento))

if (años_de_pertenencia >= 10) and (n_hijos >= 2):
    print("APROBADO")
elif (estado_civil == "C") and (n_hijos > 3) and (edad >= 45) and (edad <= 55):
    print("APROBADO")
elif (ingreso > 2500000) and (estado_civil == "S") and (residencia == "U"):
    print("APROBADO")
elif (ingreso >3500000) and (años_de_pertenencia >= 5):
    print("APROBADO")
elif (residencia == "R") and (estado_civil == "C") and (n_hijos < 2):
    print("APROBADO")
else:
 print("RECHAZADO")