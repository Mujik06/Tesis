#Nota final
PT=float(input("Ingrese la nota de su tarea"))
PI=float(input("Ingrese la nota de sus interrogaciones"))
NE=float(input("Ingrese la nota de su examen"))
PP=float(input("Ingrese la nota de su presentacion"))
desarrollo=0.3*PT+0.3*PI+0.3*NE+0.1*PP
respuesta=round(desarrollo,1)
print("Su resultado es:",respuesta)