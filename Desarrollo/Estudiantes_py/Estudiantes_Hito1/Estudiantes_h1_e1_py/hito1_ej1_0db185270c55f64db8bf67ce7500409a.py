#Nota final
pt=float(input("Ingrese nota tareas: "))
pi=float(input("Ingrese nota interrogaciones: "))
ne=float(input("Ingrese nota examen: "))
pp=float(input("Ingrese nota presentación: "))
promedio=round(0.3*pt+0.3*pi+0.3*ne+0.1*pp,1)
print(promedio)