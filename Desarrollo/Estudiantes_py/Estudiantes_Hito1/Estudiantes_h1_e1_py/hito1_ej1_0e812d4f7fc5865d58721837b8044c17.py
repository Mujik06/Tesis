#Nota final

pt = float(input("Ingresa el puntaje de Tareas: "))
pi = float(input("ingresa el puntaje de Interrogaciones: "))
ne = float(input("Ingresa el puntaje de Examen: "))
pp = float(input("Ingresa el puntaje de Presentacion: "))

promedio_final = float((0.3 * pt) + (0.3 * pi) + (0.3 * ne) + (0.1 * pp))
promedio_redondeado = round (promedio_final, 1)

print("El promedio final es :",promedio_redondeado)