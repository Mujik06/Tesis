#Aprobación de créditos
print("Hola, necesitamos que, a continuación, ingrese los datos que se le solicitan")
ingr=int(input("Sus ingresos mensuales (en pesos) "))
birth=int(input("Su año de nacimiento "))
hijos=int(input("Cuantos hijos tiene "))
perten=int(input("Los años que lleva con nuestro banco "))
civil=input("Su estado civil (S: Soltero, C: Casado) ")
casa=input("Y por último, si vive en casa o campo (U: Urbano, R: Rural) ")
edad=2021-birth
if(perten>10 and hijos>=2):
    a="Sí"
if(civil=="C" and hijos>3 and edad>=45 or edad<=55):
    a="Sí"
if(ingr>2500000 and civil=="S" and casa=="U"):
    a="Sí"
if(ingr>3500000 and perten>5):
    a="Sí"
if(casa=="R" and civil=="C" and hijos<2):
    a="Sí"
if(a=="Sí"):
    print("CRÉDITO APROBADO")
else:
    print("CRÉDITO RECHAZADO")      