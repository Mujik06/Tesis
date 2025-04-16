ingreso = int(input("Escriba sus ingresos en pesos: "))
año_nacimiento = int(input("¿Cuál es su año de nacimiento? "))
hijos = int(input("¿Cuántos hijos tiene? "))
años_pertenencia = eval(input("¿Cuántos años ha pertenecido al banco? "))
estado_civil = input("Ingrese su estado civil poniendo S si está soltero o C si está casado: ")
vivienda = input("Escriba U si vive en un lugar urbano, o R si el lugar es rural: ")

if (años_pertenencia > 10) or (hijos >= 2):
    print("APROBADO")

elif (estado_civil == "C") or (hijos > 3) or (año_nacimiento >= 1966 and año_nacimiento <= 1976):
    print("APROBADO")

elif (ingreso > 2500000) or (estado_civil == "S") or (vivienda == "C"):
    print("APROBADO")

elif (ingreso > 3500000) or (años_pertenencia > 5):
    print("APROBADO")

elif (vivienda == "C") or (estado_civil == "C") or (hijos < 2):
    print("APROBADO")

else:
    print("RECHAZADO")