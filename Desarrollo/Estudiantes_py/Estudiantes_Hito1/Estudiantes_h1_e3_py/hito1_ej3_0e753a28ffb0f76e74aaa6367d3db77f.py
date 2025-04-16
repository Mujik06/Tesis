#Aprobación de créditos
#Aprobación de créditos
año_actual=2020
ingreso=(int(input("¿Cuál es su ingreso mensual? \n")))
año=int(input("Ingrese su año de nacimiento \n"))
hijos=int(input("Ingrese número de hijos \n"))
permanencia=int(input("Ingrese los años de permanencia en el Banco \n"))
estado=input("Ingrese su estado Civil (S)=Soltero ; (C)= Casado \n")
lugar=input("Ingrese locación de su domicilio, ingrese (U)=Urbano ; (R)= Rural \n")
if(año_actual-año)>10 and hijos >=2:
  print("APROBADO")
elif estado=="C" and hijos>3 and (año_actual-año)>45 and (año_actual-año)<55:
  print("APROBADO")
elif ingreso>2500000 and estado=="S" and lugar=="R" :
  print("APROBADO")
elif ingreso>3500000 and permanencia>5 :
  print("APROBADO")
elif lugar=="R" and estado =="C" and hijos<2 :
  print("APROBADO")
else:
    print("RECHAZADO")