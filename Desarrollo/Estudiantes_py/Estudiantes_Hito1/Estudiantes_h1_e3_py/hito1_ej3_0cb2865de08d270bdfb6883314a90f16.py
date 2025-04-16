ingreso= int(input("ingrese sus ingresos en pesos: "))
fecha= input("ingrese su año de nacimiento: ")
hijos= int (input("ingrese cuantos hijos tiene: "))
años= int(input("ingrese cuantos años lleva en el banco: "))
estado=input("ingrese su estado civil: ")
casa=input("ingrese si vive en el campo o la ciudad: ")
C= estado
S= estado
R=casa
U=casa
if años>10 and hijos>=2:
    print("APROBADO")
elif años<10 and hijos<2:
    print("RECHAZADO")
elif estado == C and hijos>=3 and fecha in range(1976,1966):
    print("APROBADO")
elif estado != C and hijos<3 and fecha in range(1976, 1966) != fecha:
    print("RECHAZADO")
elif ingreso>=2500000 and estado == S and casa == U:
    print("APROBADO")
elif ingreso<2500000 and estado != S and casa != U:
    print("RECHAZADO")
elif ingreso>=3500000 and años>5:
    print("APROBADO")
elif ingreso< 3500000 and años<5:
    print("RECHAZADO")
elif casa == R and estado == C and hijos<=2:
    print("APROBADO")
elif casa != R and estado != C and hijos>2:
    print("RECHAZADO")    