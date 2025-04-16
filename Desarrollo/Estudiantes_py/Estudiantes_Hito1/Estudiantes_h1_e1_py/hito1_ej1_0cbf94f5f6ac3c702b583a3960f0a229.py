#Nota final
PT = eval(input("Ingrese el promedio de sus tareas -> "))
PI = eval(input("Ingrese el promedio de sus interrogaciones -> "))
NE = eval(input("Ingrese su nota de exámen -> "))
PP = eval(input("Ingrese nota de la presentación -> "))

PF = ((0.3 * PT) + (0.3 * PI) + (0.3 * NE) + (0.1 * PP))

aprox = round(PF, 1)

print("Su promedio final fue de", aprox)