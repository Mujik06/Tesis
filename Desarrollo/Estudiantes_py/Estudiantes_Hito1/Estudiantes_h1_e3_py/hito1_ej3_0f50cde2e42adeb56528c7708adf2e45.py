i=int(input("Ingresos mensuales>:$"))
n=int(input("Año de nacimiento>:"))
h=int(input("Nª de hijos>:"))
ab=int(input("Años siendo cliente del banco>:"))
ec=input("Estado civil C(casado)/S(soltero)>:")
v=input("Vives en zona rural(C)/urbana(U)>:")

#DATOS

e=2020-n

#PROGRAMA

if(ab>10 and h>=2):
  print("APROBADO")
elif(ec==1 and h>3 and e>=45 and e<=55):
  print("APROBADO")
elif(i>2500000 and ec=="S" and v=="U"):
  print("APROBADO")
elif(i>3500000 and ab>5):
  print("APROBADO")
elif(v=="R" and ec=="C" and h<2):
  print("APROBADO")
else:
  print("RECHAZADO")