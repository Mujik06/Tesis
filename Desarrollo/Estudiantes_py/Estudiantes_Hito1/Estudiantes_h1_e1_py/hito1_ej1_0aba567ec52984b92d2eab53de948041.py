#Nota final
Tareas = float(input("colocar nota de Tareas: "))
Interrogacion = float(input("colocar nota de Interrogacion: "))
Examen = float(input("colocar nota del Examen: "))
Presentacion = float(input("colocar nota de Presentacion : "))

notaF = 0.3*Tareas + 0.3*Interrogacion + 0.3*Examen + 0.1*Presentacion

print("La Nota Final es: ", round(notaF,1))