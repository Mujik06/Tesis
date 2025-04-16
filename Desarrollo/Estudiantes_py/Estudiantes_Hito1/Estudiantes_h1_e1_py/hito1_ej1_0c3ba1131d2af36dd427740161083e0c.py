#Nota final
a=float(input("ingrese su promedio final de las tareas: "))
b=float(input("ingrese su promedio final de las interrogaciones: "))


         

         
c=float(input("ingrese su nota del examen: "))
         
d=float(input("ingrese su nota de la presentacion: "))
NT=round((a*0.3),1)
NI=round((0.3*b),1)
NE=round((0.3*c),1)
NP=round((0.1*d),1)

print(round((NT+NI+NE+NP),1))

