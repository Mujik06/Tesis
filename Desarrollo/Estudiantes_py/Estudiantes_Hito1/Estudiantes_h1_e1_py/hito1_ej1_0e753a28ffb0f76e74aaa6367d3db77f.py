#Nota final
PT=eval(input("Ingresa nota de las tareas \n"))
PI=eval(input("Ingresa nota de las Interrogaciones \n"))
NE=eval(input("Ingresa nota del Exámen \n"))
PP=eval(input("Ingresa nota de presentación \n"))
#calcula promedio
promedio=round((0.3*PT) + (0.3*PI) + (0.3*NE) + (0.1* PP),1)
print ("Su promedio es: ", promedio)