#Nota final
import math
PT = float(input("Nota tareas:"))
PI = float(input("Nota interrogaciones:"))
NE = float(input("Nota examen:"))
PP = float(input("Nota presentación:"))
a=(0.3*PT)
b=(0.3*PI)
c=(0.3*NE)
d=(0.1*PP)
PF = round(a+b+c+d,1)
print("Promedio final:",PF)

