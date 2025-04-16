#Nota final
Val_Tarea = float
Val_Interro = float
Val_Ne= float
Val_PP= float
Val_Promedio= float

Val_Tarea = float(input("Ingrese Nota tarea : "))
Val_Interro = float(input("Ingrese Nota Interrogación : "))
Val_Ne = float(input("Ingrese Nota NE : "))
Val_PP = float(input("Ingrese Nota Presentación: "))


Val_Tarea = round((Val_Tarea * 0.30),1)
Val_Interro = round((Val_Interro * 0.30),1)
Val_Ne = round((Val_Ne * 0.30),1)
Val_PP = round((Val_PP * 0.10),1)

Val_Promedio= round((Val_Tarea+Val_Interro+Val_Ne+Val_PP), 1)
print("Promedio es : ", format(Val_Promedio))