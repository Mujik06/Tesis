#Aprobación de créditos
ingreso=int(input("indique su ingreso mensual: "))
ano_naci=int(input("ingrese su año de nacimiento: "))
n_hijo=int(input("ingrese la cantidad de hijos: "))
ano_banco=int(input("¿cuando años lleva en esta banco?: "))
es_civil=str(input("indique su estado civil: S para Soltero y C para casado: "))
lugar=str(input("indique su lugar de alojamiento: U para Urbano y R para Rural: "))
C=1
S=2
R=3
U=4
if(ano_banco>10 and n_hijo>=2):
  print("APROBADO")
elif(es_civil=="C" and n_hijo>3 and 45>=2020-ano_naci<=55):
  print("APROBADO")
elif(ingreso>=2500000 and es_civil=="S" and lugar=="U"):
  print("APROBADO")
elif(ingreso>=3500000 and ano_banco>=5):
  print("APROBADO")
elif(lugar=="R" and es_civil=="C" and n_hijo<2):
  print("APROBADO")
else:
  print("RECHAZADO")      