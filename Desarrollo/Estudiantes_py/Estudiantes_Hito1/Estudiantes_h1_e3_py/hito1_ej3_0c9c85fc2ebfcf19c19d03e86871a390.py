#Aprobación de créditos
ingreso = int(input("Ingrese valor de ingreso mensual: "))
nacimiento = int(input("Ingrese año nacimiento: "))
nrohijos = int(input("Ingrese numero hijos: " ))
pertbanco = int(input("Ingrese años perteneciendo al banco: "))
estadocivil = input("Ingrese estado civil, S: Soltero o C: Casado: ")
campociudad = input("Vive en campo o ciudad? Ingrese U: Urbano o R: Rural: ")

edad= int(2021-nacimiento)

if pertbanco>10 and nrohijos>=2:
    print("APROBADO")
elif (estadocivil== "S" or estadocivil == "s") and nrohijos>3 and (edad > 44 and edad < 55):
    print("APROBADO")
elif ingreso> 2500000 and (estadocivil=="s" or estadocivil=="S") and (campociudad=="C" or campociudad == "c"):
    print("APROBADO")
elif ingreso>3500000 and pertbanco>5:
    print("APROBADO")
elif campociudad=="R" and estadocivil=="C" and nrohijos<2:
    print("APROBADO")
else:
    print("RECHAZADO")