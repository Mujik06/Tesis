#Nota final
# Entradas

PT = float(input("Ingresa el promedio de notas de las tareas: "))
PI = float(input("Ingresa el promedio de notas de las interrogaciones: "))
NE = float(input("Ingresa la nota de tu examen: "))
PP = float(input("Ingresa la nota de tu presentacion: "))

# Calculos

if 1 <= PT and PI and NE and PP <= 7:

    nota_final = 0.3*PT + 0.3*PI + 0.3*NE + 0.1*PP

    nota_final = round(nota_final,1)

    print("Tu promedio es el siguiente",nota_final)

else:
    print("No podemos calcular favor Ingresa notas entre 1 y 7")