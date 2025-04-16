#Aprobación de créditos
from datetime import date

IG = int(input('Ingrese ingresos (en pesos): '))
BY = int(input('Ingrese año de nacimiento: '))
NH = int(input('Ingrese numero de hijos: '))
YP = int(input('Ingrese años de pertenencia al banco: '))
EC = input('Ingrese estado civil (S: soltero, C: casado: ')
CC = input('Ingrese zona de residencia (U: urbano, R: rural: ')

HD = date.today()
YY = HD.year - BY

#Si el cliente pertenece más de 10 años al banco, y tiene dos o más hijos
if YP > 10 and NH >= 2:
  print('APROBADO')
#Si el cliente es casado, tiene más de tres hijos, y tiene entre 45 y 55 años
elif EC == 'C' and NH > 3 and YY >= 45 and YY <= 55:
  print('APROBADO')
#Si el cliente posee ingresos superiores a $2.500.000, es soltero y vive en la ciudad
elif IG > 2500000 and EC == 'S' and CC == 'U':
  print('APROBADO')
#Si el ciente tiene ingresos superiores a $3.500.000 y pertenece al banco por más de 5 años
elif IG > 3500000 and YP > 5:
  print('APROBADO')
#Si el cliente vive en el campo, es casado y tiene menos de dos hijos
elif CC == 'R' and EC == 'C' and NH < 2:
  print('APROBADO')
else:
  print('RECHAZADO')






