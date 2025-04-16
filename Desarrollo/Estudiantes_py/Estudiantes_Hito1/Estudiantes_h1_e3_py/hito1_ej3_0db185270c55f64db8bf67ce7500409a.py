#Aprobación de créditos
a=int(input("Ingreso en peso: "))
b=int(input("Año de nacimiento: "))
c=int(input("Numero de hijos: "))
d=int(input("Año pertenencia al banco: "))
e=str(input("Estado civil Solero,S o Casado, C: "))
f=str(input("Vive en campo o ciudad("U": urbano, "R": rural): "))

if d>10 and c>=2:
    print("APROBADO")
elif e=="C" and c>3 and 45<=(2021-f)>=55:
    print("APROBADO")
elif a>2500000 and e=="S" or e=="s" and f=="U" or f=="u":
    print("APROBADO")
elif a>3500000 and d>5:
    print("APROBADO")
elif f=="R" and e=="C" or e=="c" and c<2:
    print("APROBADO")
else:
    print("RECHAZADO")
