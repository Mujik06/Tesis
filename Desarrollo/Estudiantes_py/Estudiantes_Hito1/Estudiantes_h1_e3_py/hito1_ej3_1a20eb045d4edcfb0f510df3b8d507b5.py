Ingreso= int(input("ingrese su ingreso: "))
Año_de_nacimiento= int(input("ingrese su año de nacimiento: "))
Numero_de_hijos= int(input("ingrese su numero de hijos :"))
Años_de_pertenencia= int(input("ingrese sus años de pertenencia en nuestro banco: "))
Estado_civil= input("ingrese su estado civil con S o C ")
vive= input("seleccione su localidad U o R ")

if Años_de_pertenencia>=10:
    if Numero_de_hijos>=2:
        print("APROBADO")
if Estado_civil=="C":
    if Numero_de_hijos>=3:
        if Año_de_nacimiento-2021>45 and Año_de_nacimiento-2021<=55:
            print("APROBADO")
if Ingreso*12> 2500000:
    if Estado_civil=="S" and vive=="U":
        print("APROBADOO")

if Ingreso*12> 3500000:
    if Años_de_pertenencia>=5:
        print("APROBADO")
if vive=="R" and Estado_civil=="C":
    if Numero_de_hijos<=2:
        print("APROBADO")
