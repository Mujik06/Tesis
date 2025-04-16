ing=float(input("Ingrese sus ingresos: $"))
año=int(input("Año de nacimiento:"))
numeroh=int(input("Numero de hijos:"))
añosba=int(input("Años de pertenencia al banco:"))
esta=input("Estado Civil\n(S)Soltero\n(C)Casado:")
vive=input("Residencia\n(U) Urbano\n(R)Rural:")

edad=2021-año

ciu="U"
camp="R"
C="C"
S="S"
if (añosba>=10 and numeroh>=2) or (esta==C and numeroh<=3 and edad<=45 and edad<=55) or (ing>=2500000 and esta==S and vive==ciu) or (ing>=3500000 and añosba>= 5) or(vive==camp and esta==C and numeroh<=2):
    
    print("APROBADO")

else:
     print("RECHAZADO")