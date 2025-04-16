#Nota final
PT = float(input('Ingresa nota de Tarea: '))
PI = float(input('Ingresa nota de Tarea: '))
NE = float(input('Ingresa nota de Tarea: '))
PP = float(input('Ingresa nota de Tarea: '))

Promedio  = (0.3 * PT + 0.3 * PI + 0.3 * NE + 0.1 * PP)
Promedio_Redondeado = round(Promedio,1)
print(Promedio_Redondeado)