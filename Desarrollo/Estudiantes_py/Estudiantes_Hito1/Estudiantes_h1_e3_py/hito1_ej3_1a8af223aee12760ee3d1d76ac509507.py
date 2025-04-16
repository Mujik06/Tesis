#Aprobación de créditos
ingreso = eval(input("Cantidad de ingresos (En pesos): "))
año_nacimiento = eval(input("Ingrese su año de nacimiento: "))
num_hijos = eval(input("Igrese la cantidad de hijos que tiene: "))
años_banco = eval(input("Ingrese los años de pertenencia en el banco: "))
estado_civil = input("Ingrese su estado civil(responda con 'S' si está soltero/a o 'C' si está casado/a: ")
vivienda = input("Ingrese donde vive (responda con una 'U' si es una zona urbana o 'R' si es una zona rural: ")
año_actual = 2021
edad = año_actual - año_nacimiento
if (años_banco > 10 and num_hijos >= 2):
  print("APROBADO")
elif (estado_civil == "C" and num_hijos > 3 and edad > 45 and edad < 55):
    print("APROBADO")
elif (ingreso > 2500000 and estado_civil == "S" and vivienda == "U"):
    print("APROBADO")
elif (ingreso > 3500000 and años_banco > 5):
  print("APROBADO")
elif (vivienda == "R" and estado_civil == "C" and num_hijos < 2):
   print("APROBADO")
else:
 print("RECHAZADO")    