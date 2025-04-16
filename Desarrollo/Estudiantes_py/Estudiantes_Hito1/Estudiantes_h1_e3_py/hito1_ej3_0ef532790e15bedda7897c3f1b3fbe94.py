#Aprobación de créditos
a=int(input("Ingreso en pesos:"))
b=int(input("Año de nacimiento:"))
c=int(input("Numero de hijos"))
d=int(input("Años de pertenencia al banco"))
e=(input("Estado civil:\ nC) Casado \nS) Soltero"))
f=(input("Urbano o rural:\nU) Urbano \nR) Rural"))

if (d > 10 and c > 2):
   print("APROBADO")
elif(e == "C" and c > 3 and 1961 < b < 1971):
   print("APROBADO")
elif (a > 2500000 and e == "S" and f == U):
   print("APROBADO")
elif (a > 3500000 and d > 5):
   print("APROBADO")
elif (f == "R" and c < 2):
   print("APROBADO")
else:
   print("RECHAZADO")
    
