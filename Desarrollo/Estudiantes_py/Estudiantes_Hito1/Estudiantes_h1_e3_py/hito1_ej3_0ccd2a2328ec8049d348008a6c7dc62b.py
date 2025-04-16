im=int(input("Ingresos mensuales>:$"))
an=int(input("Año de nacimiento>:"))
nh=int(input("Número de hijos>:"))
ab=int(input("Años siendo cliente del banco>:"))
sc=input("Estado Civil C(casado)/S(soltero)>:")
ru=input("Vives en zona Rural(R)/Urbana(U)>:")
x=2020-an
if(ab>10 and nh>=2):
  print("APROBADO")
elif(sc==1 and nh>3 and x>=45 and x<=55):
  print("APROBADO")
elif(im>2500000 and sc=="S" and ru=="R"):
  print("APROBADO")
elif(im>3500000 and ab>5):
  print("APROBADO")
elif(ru=="R" and sc=="C" and nh<2):
  print("APROBADO")
else:
  print("RECHAZADO")