#Nota final
PT = float(input("Ingrese la nota de sus tareas:"))
PI = float(input("Ingrese la nota de las interrogaciones:"))
NE = float(input("Ingrese su nota del examen:"))
PP = float(input("Ingrese su nota de presentación:"))

promedio_final = 0.3*PT + 0.3*PI + 0.3*NE + 0.1*PP

print("La nota final es:", round(promedio_final,1))