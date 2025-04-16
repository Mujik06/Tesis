#Nota final con float : Decimales.

PT = int and float(input("Ingrese su nota de Tareas - Equivale 30%: "))
PI = int and float(input("Ingrese su nota de Interrogaciones - Equivale 30%: "))
NE = int and float(input("Ingrese su nota de Examen - Equivale 30%: "))
PP = int and float(input("Ingrese su nota de Presentación - Equivale 10%: "))

nota_pt = PT*0.3
nota_pi = PI*0.3
nota_ne = NE*0.3
nota_pp = PP*0.1

# Acá se realiza el cáculo
nota_promedio = PT + PI + NE + PP 
nota_ponderada = nota_pt + nota_pi + nota_ne + nota_pp
nota_final = nota_promedio / nota_ponderada

# El resultado de la variable y la tomo y la dejo con un decimales
calculo = round(nota_final,1)
print('El promedio: ', calculo)