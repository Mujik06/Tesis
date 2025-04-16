#Aprobacion de creditos
I=int(input("Ingreso en pesos: "))
N=int(input("Año de nacimiento: "))
H=int(input("Numero de hijos: "))
P=int(input("Años de pertenencia al banco: "))
E=input("Estado civil (S:solter,C:casado): ")
V=input("En donde vive (U:urbano,R:rural): ")
A=2016-N
if P>10 and H>=2:
    print("APROBADO")
elif I>2500000 and E=="S" and V=="U":
    print("APROBADO")
elif 55>=A>=45 and E=="C" and H>3:
    print("APROBADO")
elif I>3500000 and P>5:
    print("APROBADO")
elif V=="R" and E=="C" and H<2:
    print("APROBADO")
else:
    print("RECHAZADO")