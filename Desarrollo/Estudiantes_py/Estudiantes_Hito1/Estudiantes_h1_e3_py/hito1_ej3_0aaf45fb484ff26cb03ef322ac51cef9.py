#Aprobación de créditos
ing = eval(input("Ingreso:"))
nac = eval(input("Año de nacimiento:"))
hijos = eval(input("Número de hijos:"))
pert = eval(input("Años de pertenencia al banco:"))
estado = input("(S:soltero,C:casado):")
vive = input("(U:urbano,R:rural):")
      
if pert > 10 and hijos >= 2:
 print("APROBADO")
elif estado == "C" and hijos > 3 and nac >= 1965 and nac <= 1975:
 print("APROBADO")
elif ing > 2500000 and estado == "S" and vive == "U":
 print("APROBADO")
elif ing > 3500000 and pert > 5:
 print("APROBADO")
elif vive == "R" and estado == "C" and hijos < 2:
 print("APROBADO")
else:
 print("REPROBADO")