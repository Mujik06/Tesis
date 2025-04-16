#Aprobación de créditos
i=int(input("ingrese su ingreso en pesos: "))
an=int(input("ingrese su año de nacimiento: "))
n=int(input("ingrese numero de hijos: "))
ab=int(input("ingrese los años de pertenencia al banco: "))
ec=input("Estado civil: ")
v=input("ingrese si vive en campo o ciudad: ")
if 10<ab and 2<=n:
 print("APROBADO")
else:
   print("RECHAZADO")
if ec=="C" and n>3 and 45<=(2018-an) and   (2018-an)<=55:
 print("APROBADO")
else:
   print("RECHAZADO")
if 2500000<i and ec=="S" and v=="U":
 print("APROBADO")
if 3500000<i and 5<ab:
 print("APROBADO")
if v=="R" and ec=="C" and n<2:
 print("APROBADO")