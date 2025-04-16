#Aprobación de créditos
i=int(input("Introduzca sus ingresos: "))
an=int(input("Introduzca su año de nacimiento: "))
nh=int(input("Introduzca su numero de hijos: "))
ap=int(input("Hace cuantos años pertenece al banco: "))
ec=str(input("Estado civil, si es casado, introduzca C, si es soltero, S: "))
v=str(input("Si vive en campo, introduzca R, si vive en ciudad, U: "))
if ap>10 and nh>2:
    print("APROBADO")
elif i>2500000 and ec=='S' and v=='U':
    print("APROBADO")
elif i>3500000 and ap>5:
    print("APROBADO")
elif v=='R' and ec=='C' and 0<=nh<2:
    print("APROBADO")
elif ec=='C' and nh>3 and an>1962 and an<1972:
    print("APROBADO")
else:
    print("RECHAZADO")