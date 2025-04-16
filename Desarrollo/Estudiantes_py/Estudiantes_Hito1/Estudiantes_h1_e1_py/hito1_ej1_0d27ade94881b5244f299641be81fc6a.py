#Nota final
print("ingrese las 4 notas")
Ntareas = eval(input())
Ninterrogantes = eval(input())
Nexamen = eval(input())
Npresentacion = eval(input())
Npromedio = (0.3 * Ntareas + 0.3 * Ninterrogantes + 0.3 * Nexamen + 0.1 * Npresentacion) 
Npromedio1 = round(Npromedio,1)
print(Npromedio1)