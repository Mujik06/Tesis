#Nota final
pt = float(input("ingrese nota tareas: "))
pi = float(input("ingrese nota interrogaciones: "))
ne = float(input("ingrese nota examen: "))
pp = float(input("ingrese nota presentacion: "))

nf = pt * 0.3 + pi * 0.3 + ne * 0.3 + pp * 0.1
#float(nf)

redondeado = round(nf, 1)
print(redondeado)