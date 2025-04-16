#Aprobación de créditos
s="s"
c="c"
u="u"
r="r"
I=int(input("Ingrese el valor de sus ingresos:"))
A=int(input("Ingrese su edad:"))
H=int(input("Ingrese su cantidad de hijos:"))
i=int(input("Ingrese años de lleva afiliado al banco:"))
print("ESTADO CIVIL")
print("S-Soltero")
print("C-Casado")
EC=input("Ingrese su opción de Estado Civil:").lower()
print("¿DONDE VIVE?")
print("U-Urbano")
print("R-Rural")
CUI=input("Ingrese su opción de vivienda:").lower()

if (i>=10) and (H>=2):
    print("APROBADO")
elif EC in c and (H>=3) and (A>=1965) and (A<=1975):
    print ("APROBADO")
elif (I>=2500000) and EC in s and CUI in r:
    print("APROBADO")
elif (I>=3500000) and (i>=5):
    print("APROBADO")
elif CUI in r and EC in c and (H<=2):
    print ("APROBADO")
else:
    print("RECHAZADO")