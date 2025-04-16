#Aprobación de créditos
ingreso = int(input("Ingrese su ingreso (en pesos): "))
año_nacimiento = int(input("Ingrese su año de nacimiento: "))
numero_hijos = int(input("Ingrese su numero de hijos: "))
años_pertenencia = int(input("Ingrese cuantos años lleva con el banco: "))
estado_cilvil = input("Ingrese su estado civil (""S"": soltero, ""C"": Casado): ")
lugar = input("Ingrese si vive en campo o ciudad (""U"": Urbano, ""R"": Rural): ")

if años_pertenencia >= 10 and numero_hijos >= 2:
    print("APROBADO")
elif (estado_cilvil == "C" or estado_cilvil == "c") and numero_hijos  > 3 and 1976 <= año_nacimiento <= 1986:
    print("APROBADO")
elif ingreso > 2500000 and (estado_cilvil == "S" or estado_cilvil == "s") and (lugar == "U" or lugar == "u"):
    print("APROBADO")
elif ingreso > 3500000 and años_pertenencia > 5:
    print("APROBADO")
elif (lugar == "r" or lugar == "R") and (estado_cilvil == "C" or estado_cilvil == "c") and numero_hijos < 2:
    print("APROBADO")
else:
    print("RECHAZADO")