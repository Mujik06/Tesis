#Aprobación de créditos
a=int(input("Ingreso en pesos:"))
b=int(input("Año de nacimiento:"))
n=int(input("Número de hijos:"))
x=int(input("Años de pertenencia al banco:"))
y=input("Estado civil, S si es soltero o C si es casado:")
z=input("Si vive en campo o ciudad(U para urbano y R para rural):")

p=(1966, 1967,1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976)

if x > 10 and n >= 2:
    print("APROBADO")
else:
    if y == "C" and n > 3 and b == p:
        print("APROBADO")
    else:
        if a>2500000 and y=="S" and z=="U":
            print("APROBADO")
        else:
            if a > 3500000 and x > 5:
                print("APROBADO")
            else:
                if z=="R" and y=="C" and n<2:
                    print("APROBADO")
                else:
                    print("RECHAZADO")     