#Nota final
PT = float(input("Ingrese promedio de tareas: "))
PI = float(input("Ingrese promedio de interrogaciones: "))
NE = float(input("ingrese nota del examen: "))
PP = float(input("ingrese promedio de presentacion: "))

PF = (0.3*(PT))+(0.3*(PI))+(0.3*(NE))+(0.1*(PP))
a = round(PF,1)
print(a)
