#Nota final
pt = float(input("Ingresa tu nota de tareas: "))
pi = float(input("Ingresa tu nota de interrogaciones: "))
ne = float(input("Ingresa tu nota del examen: "))
pp = float(input("Ingresa tu nota de presentacion: "))

pf = round(0.3*pt + 0.3*pi + 0.3*ne + 0.1*pp,1)

print("Tu promedio final es ", pf)