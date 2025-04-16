#Nota final
PT=float(input("ingrese nota de tareas: "))
PI=float(input("ingrese nota de interrogaciones: "))
NE=float(input("ingrese nota de examen: "))
PP=float(input("ingrese nota de presentacion: "))
PF=float((PT*0.3)+(PI*0.3)+(NE*0.3)+(PP*0.1))
print("el promedio final es: ", round(PF,1))