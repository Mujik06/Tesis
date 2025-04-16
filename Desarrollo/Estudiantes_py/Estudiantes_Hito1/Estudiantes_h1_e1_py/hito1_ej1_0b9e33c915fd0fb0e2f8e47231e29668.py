#Nota final
PT=float(input("Ingrese promedio de tareas: "))
PI=float(input("Ingrese promedio de interrogaciones: "))
NE=float(input("Ingrese nota del examen: "))
PP=float(input("Ingrese nota de presentacion: "))
PF=(PT*30+PI*30+NE*30+PP*10)/100
print("promedio final: ",PF)