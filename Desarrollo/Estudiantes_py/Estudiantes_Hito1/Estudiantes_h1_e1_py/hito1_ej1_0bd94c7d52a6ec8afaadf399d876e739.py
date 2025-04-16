#Nota final
# Entradas --------
PT = float(input("Ingresa la nota de tareas: "))
PI  = float(input("Ingresa la nota de interrogaciones: "))
NE = float(input("Ingresa la nota de examén: "))
PP = float(input("Ingresa la nota de presentaciones: "))

# Variables-------------------------------------------------------
NF = (float((0.3*PT) + (0.3*PI) + (0.3*NE) + (0.1*PP)))

# Salida--------------------------------------------------------------------------
print("La nota final es:", round(NF,1))
     