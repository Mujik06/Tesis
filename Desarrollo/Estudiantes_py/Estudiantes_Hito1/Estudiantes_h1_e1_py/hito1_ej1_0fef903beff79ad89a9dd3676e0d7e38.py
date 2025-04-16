#Nota final
PT=float(input("Ingrese la nota de las tareas:"))
PI=float(input("Ingrese la nota de las interrogaciones:"))
NE=float(input("Ingrese la nota del examen:"))
PP=float(input("Ingrese la nota de presentacion:"))
PT=PT*0.3
PI=PI*0.3
NE=NE*0.3
PP=PP*0.1
NF=PT+PI+NE+PP
print("La nota final es:",NF)