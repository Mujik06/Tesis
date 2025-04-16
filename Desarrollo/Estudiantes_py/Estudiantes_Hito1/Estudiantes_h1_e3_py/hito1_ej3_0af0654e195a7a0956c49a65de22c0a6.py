#Aprobación de créditoS
ingreso = int(input("Escriba su ingreso mensual en pesos"))
byear = int(input("Escriba su año de nacimiento"))
nhijos = int(input("¿Cuantos hijos tiene?"))
apb = int(input("¿Hace cuantos años es cliente del banco?"))
ec = input("Ingrese su estado civil (S o C)")
vive = input("Ingrese su lugar de residencia (Urbano o Rural")
edad = int(2017 - byear)
if apb >= 10 and nhijos >= 2:
    print("APROBADO")
else:
    if ec == "C" and nhijos > 3 and edad > 45 and edad < 55:
        print("APROBADO")
    else:
        if ingreso >= 2500000 and ec == "S" and vive == "U":
            print("APROBADO")
        else:
            if ingreso >= 3500000 and apb >= 5:
                print("APROBADO")
            else:
                if ec == "C" and vive == "R" and nhijos <= 2:
                    print("APROBADO")
                else:
                    print("RECHAZADO")
    
