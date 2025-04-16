#Aprobación de créditos
print("estaso civil: s si es soltero/a o c si es casado/a")
print("ingrese u si vive en la ciudad o ingrese r si vive en el campo cuando se le pregunte")
ingreso=int(input("ingrese su ingreso:"))
ano_de_nacimiento=int(input("ano de nacimiento:"))
edad=2018-ano_de_nacimiento
numero_hijos=int(input("cuantos hijos tiene:"))
anos_de_pertenencia_al_banco=int(("anos de pertenencia al banco:"))        
estado_civil=int(input("estado civil:"))                     
vivienda= int(input("¿vive en el campo o en la ciudad?"))

if anos_de_pertenencia_al_banco>10 and numero_hijos>=2:
  print("APROBADO")
if estado_civil==c and numero_hijos>=3 and 55>=edad>=45:
  print("APROBADO")
if ingreso>2500000 and  estado_civil==s and vivienda==u:
  print("APROBADO")
if ingreso>3500000 and anos_de_pertenencia_al_banco>5:
  print("APROBADO")
if vivienda==r and estado_civil==c and numero_hijos<2:
  print("APROBADO")
else:
  print("RECHAZADO")

                    

