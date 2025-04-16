#Nota final
PT = float(input("Ingrese el promedio de notas de sus trabajos: "))
PI = float(input("Ingrese el promedio de notas de sus interrogaciones: "))
NE = float(input("Ingrese su nota de examen: "))
PP = float(input("Ingrese el promedio de sus presentaciones: "))

promedio = (0.3*PT+0.3*PI+0.3*NE+0.1*PP)

print("Su promedio Final es", round(promedio,1))