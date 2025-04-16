#Aprobación de créditos
 
from datetime import date
from datetime import datetime
x=int(input("ingreso"))
y=int(input("año de nacimiento"))
z= date.today()
edad= z.year-y
Numh=int(input("numero de hijos"))
aper=int(input("años permanencia al banco"))
estadocivil=input("S si es soltero y C si es casado")
ubicacion=input("U: urbano, R: rural")
if aper>10 and Numh>=2:
  print("APROBADO")
elif estadocivil=="C" and Numh>3 and edad>45 and edad<55:
  print("APROBADO")
elif x>2500000 and estadocivil=="S" and ubicacion=="U":
  print("APROBADO")
elif x>3500000 and aper>5:
  print("APROBADO")
elif ubicacion=="R" and estadocivil=="C" and Numh<2:
  print("APROBADO")
else:
  print("RECHAZADO")    