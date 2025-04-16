#Aprobación de créditos
ingresos=int(input("ingresos(en pesos); "))
añonacimiento=int(input("año de nacimiento: "))
ndehijos=int(input("numero de hijos: "))
añosdedepertenecia=int(input("años de pertenencia al banco: "))
estadocivil=input("estado civil(S para soltero, C para casado): ")
campociudad=input("lugar de residencia(U para urbano, R para rural): ")
x=0

if(añosdedepertenecia>10):
    if(ndehijos>=2):
        x="aprobado"
       
   
if(estadocivil=="C"):
    if(ndehijos>3):
        if(1972>añonacimiento>1962):
            x="aprobado"
          
            
      
if(ingresos>2500000):
    if(estadocivil=="S"):
        if(campociudad=="U"):
            x="aprobado"
if(ingresos>3500000):
     if(añosdedepertenecia>5):
            x="aprobado"

if(ingresos>3500000):
     if(añosdedepertenecia>5):
            x="aprobado"
if(campociudad=="R"):
     if(estadocivil=="C"):
         if(ndehijos<2):
             x="aprobado"
if(x=="aprobado"):
    print("APROBADO")
else:
     print("NO APROBADO")
