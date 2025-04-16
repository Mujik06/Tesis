PT = float(input("Ingrese nota Tareas: "))
PI = float(input("Ingrese nota Interrogaciones: "))
NE = float(input("Ingrese nota Examen: "))
PP = float(input("Ingrese nota Presentaciones : "))
# PROCESAMIENTO
NF = (0.3*PT + 0.3*PI + 0.3*NE + 0.1*PP)
# SALIDA
print("La NF del curso es: ", round(NF,1))
         

