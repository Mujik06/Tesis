#Aprobación de créditos
print("ingrese sus datos para determinar la aprobacion o rechazo del credito")

ingreso = int(input("ingrese su ingresos en pesos: "))
nacimiento = int(input("ingrese su año nacimiento: "))
hijos = int(input("ingrese su numero de hijos: "))
antiguedad = int(input("ingrese su hace cuantos año esta en el banco: "))
estado = input("ingrese C si esta casado o S si esta soltero: ")
residencia = input("ingrese U si  vive en ciudad o R si vive en el campo: ")
N = 2021 - nacimiento


if antiguedad > 10 and hijos >=2:
    print("su creditos fue APROBADO")

elif estado=="C" and hijos >3 and N >=45 and N<=55:
    print("su credito fue APROBADO")

elif ingreso >2500000 and estado=="S" and residencia=="U":
    print("su credito fue APROBADO")

elif ingreso >3500000 and antiguedad > 5:
    print("su credito fue APROBADO")

elif residencia=="R" and estado=="C" and hijos <2:
    print("su credito fue APROBADO")

else:
    print("su credito fue rechazado")
