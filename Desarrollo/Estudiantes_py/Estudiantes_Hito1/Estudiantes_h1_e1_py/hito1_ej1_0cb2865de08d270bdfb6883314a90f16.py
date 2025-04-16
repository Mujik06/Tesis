#Nota final
PT= float(input("Ingrese Nota de Tareas: "))
PI= float(input("Ingrese Nota de Interrogaciones: "))
NE= float(input("Ingrese Nota de Examen: "))
PP= float(input("Ingrese Nota de Presentacion: "))

final= (PT*0.3)+(PI*0.3)+(NE*0.3)+(PP*0.1)
final=round(final,1)
print(final)      