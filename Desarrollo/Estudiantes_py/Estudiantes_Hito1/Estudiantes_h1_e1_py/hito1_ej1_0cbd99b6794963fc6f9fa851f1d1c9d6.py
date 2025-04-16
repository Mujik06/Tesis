#Nota final
      
PT= float(input("Ingrese su nota de las tareas "))
PI = float(input("Ingrese su nota de las Interrogaciones "))
NE = float(input(" Ingrese su nota del Examen "))
PP = float(input("Ingrese su nota por Presentacion "))

X= 0.3*PT + 0.3*PI + 0.3*NE + 0.1*PP

print("Su promedio final es:",round(X,1))