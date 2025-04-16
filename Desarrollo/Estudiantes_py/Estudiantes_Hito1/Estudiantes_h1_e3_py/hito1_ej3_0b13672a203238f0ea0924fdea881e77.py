P = int(input("Ingrese año de nacimiento: "))
K = int(input("Ingrese numero de hijos: "))
Q = int(input("Ingrese ingresos: "))
B = int(input("Ingrese año de pertenecia al banco: "))
U = str(input("Soltero o Casado: "))
T = str(input("Ingrese donde vive: "))

edad = 2021 - P

if B > 10 and K >= 2:
    print("APROBADO")

elif B <= 10 and K < 2:
    print("RECHAZADO")

if U == "casado" and K > 3 and edad >= 45 or edad <= 55:
    print("APROBADO")

elif U!= "casado" and U < 2 and edad < 45 or edad > 55:
    print("RECHAZADO")

if Q > 2500000 and U == "soltero" and T == "urbano":
    print("APROBADO")

elif Q < 2500000 and U != "soltero" and T != "urbano":
    print("RECHAZADO")

if Q > 3500000 and B > 5:
    print("APROBADO")

elif Q < 3500000 and B < 5:
    print("RECHAZADO")

