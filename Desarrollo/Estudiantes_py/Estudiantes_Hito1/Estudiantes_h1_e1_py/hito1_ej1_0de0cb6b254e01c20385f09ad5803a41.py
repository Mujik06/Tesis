#Nota final

pt = eval(input("Ingresa la nota de tu tarea: "))
pi = eval(input("Ingresa la nota de la interrogación: "))
ne = eval(input("Ingresa la nota de tu examen: "))
pp = eval(input("Ingresa la nota de la presentación "))

promedio = (0.3*pt) + (0.3*pi) + (0.3*ne) + (0.1*pp)
promedio_redondeado = round(promedio, 1) 

print("El promedio de tus notas es:", promedio_redondeado)