#Aprobación de créditos
#Entradas------------------------------
ingresos = int(input("Ingrese sus ingresos: "))
AN = int(input("Ingrese su año de nacimiento: "))
NH = int(input("Ingrese cuantos hijos tiene: "))
APB = int(input("Ingrese hace cuantos años pertenece al banco: "))
EC = input("Indique su estado civil (S: soltero, C: casado): ")
LV =input("Indique  donde vive (U: urbano, R: rural): ")

#Variables-------
edad = (2021 - AN)

#Condiciones---------
if APB > 10 and NH >= 2:
    print("APROBADO")

elif (EC == "C" or EC == "c") and NH > 3 and 45 <= edad <= 55:
    print("APROBADO")

elif ingresos > 2500000 and (EC == "S" or EC == "s") and (LV == "U" or LV == "u"):
    print("APROBADO")

elif ingresos > 3500000 and APB > 5:
    print("APROBADO")

elif (EC == "C" or EC == "c") and NH < 2 and (LV == "R" or LV == "r"):
    print("APROBADO")

else:
    print("RECHAZADO")