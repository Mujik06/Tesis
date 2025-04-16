#Aprobación de créditos
i= int(input("ingrese sus ingresos(clhp): "))
a=int(input("ingrece año de nacimiento: "))
n=int(input("ingrese numero de hijos: "))
b=int(input("ingrese años de pertenencia al banco: "))
e=input("ingrese estado civil (S: soltero, C: casado): ")
v= input("ingrese si vive en campo o cuidad ("U": urbano, "R": rural) : ")

if b > 10 and n>=2:
          print("APROBADO")
elif e=="c" and n >= 3 and 45 <= (2018-a) <= 55:
          print("APROBADO")
elif i >= 250000 and e =="S" and v == "U":
          print("APROBADO")
elif i >= 350000 and b >= 5:
          print("APROBADO")
elif v == "R" and e =="C" and n <= 2:
          print("APROBADO")
else:
          print("RECHAZADO")      