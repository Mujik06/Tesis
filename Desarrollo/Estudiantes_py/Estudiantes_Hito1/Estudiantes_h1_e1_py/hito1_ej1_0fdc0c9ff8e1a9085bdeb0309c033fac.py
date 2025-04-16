#Nota final

PT = float(input("Ingresar nota tareas: "))
PI = float(input("Ingresar nota interrogaciones: "))
NE = float(input("Ingresar nota examen: "))
PP = float(input("Ingresar nota presentacion: "))

NF = (0.3*PT + 0.3*PI + 0.3*NE + 0.1*PP)

print(format(NF, "0.1f"))