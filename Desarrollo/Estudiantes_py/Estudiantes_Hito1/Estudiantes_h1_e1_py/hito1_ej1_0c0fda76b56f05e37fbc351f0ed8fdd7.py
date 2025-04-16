#Nota final
PT = float(input("ingrese la nota de tus tareas : "))
PI = float(input("ingrese la nota de tus interrogaciones : "))
NE = float(input("ingrese la nota de tu examen : "))
PP = float(input("ingrese la nota de tu presentacion : "))

PromedioF = (0.3*PT) + (0.3*PI) + (0.3*NE) + (0.1*PP)

print ("Su promedio es : ",round(PromedioF, 2))    