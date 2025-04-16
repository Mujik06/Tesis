#Nota final
print("ingrese sus notas")

PT= float(input("Ingrese su nota de tarea "))
PI= float(input("Ingrese su nota de interrogaciones "))
NE= float(input("ingrese su nota de examen "))
PP= float(input("Ingrese su nota de presentacion "))

promedio=  round(0.3*PT + 0.3*PI + 0.3*NE + 0.1*PP, 1)

print(" el promedio de sus notas es : ", promedio)