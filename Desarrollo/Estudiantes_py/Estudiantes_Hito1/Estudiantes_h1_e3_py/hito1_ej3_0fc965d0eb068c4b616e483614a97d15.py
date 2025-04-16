#Aprobación de créditos
print("Bienvenido al formulario para el credito del banco.")
print("A continuación le preguntarán ciertas informaciones que debe responder.")

#Pedir los datos requeridos para luego filtrar

ingresos = int(input("Ingresos (en pesos chilenos): "))
año_de_nacimiento = int(input("Ingrese su año de nacimiento: "))
edad = 2020 - año_de_nacimiento
años_en_el_banco = int(input("¿Cuantos años lleva en el banco? "))
numero_de_hijos = int(input("Número de hijos: "))
estado_civil = input("Estado civil (S: Soltero, C: Casado) " )
zona = input("¿Vive en zona urbana o rural (U: Urbano, R: Rural)? ")

#Filtros

if años_en_el_banco > 10 and numero_de_hijos >= 2:
    print("APROBADO")
else:
    if estado_civil == "C" or "c" and numero_de_hijos > 3 and 45 <= edad <= 55:
        print("APROBADO")
    else:
        if ingresos > 2500000 and estado_civil == "S" or "s" and zona == "U" or "u":
            print("APROBADO")
        else:
            if ingresos > 3500000 and años_en_el_banco > 5:
                print("APROBADO")
            else:
                if zona == "R" and estado_civil == "C" or "c" and numero_de_hijos < 2:
                    print("APROBADO")
                else:
                    print("RECHAZADO")      