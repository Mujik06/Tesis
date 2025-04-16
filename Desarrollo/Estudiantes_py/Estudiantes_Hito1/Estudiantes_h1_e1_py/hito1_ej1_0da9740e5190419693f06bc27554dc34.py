#Nota final
PT=float(input("ingrese nota de tareas: "))
PI=float(input("ingrese nota de interrogaciones: "))
NE=float(input("ingrese nota de examen: "))
PP=float(input("ingrese nota de presentacion: "))

print(round(0.3*PT + 0.3*PI + 0.3*NE + 0.1*PP,1))
      