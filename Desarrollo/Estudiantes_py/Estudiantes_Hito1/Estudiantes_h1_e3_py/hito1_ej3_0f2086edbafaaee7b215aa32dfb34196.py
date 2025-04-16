#Aprobación de créditos
a=int(input("Ingreso:"))
b=int(input("Año de nacimiento:"))
z=int(input("Número de hijos:"))
d=int(input("Año de pertenencia al banco:"))
e=input("Estado civil:")
f=input("Si vive en campo o ciudad:")
edad = 2017-b
if d>10 and z>=2:
    print("APROBADO")
elif e=="C" and z>3 and 45<=edad<=55:
    print("APROBADO")
elif a>2500000 and e=="S" and f=="U":
    print("APROBADO")
elif a>3500000 and d>5:
    print("APROBADO")
elif f=="R" and e=="C" and z<2:
    print("APROBADO")
else:
    print("RECHAZADO")