#Aprobación de créditos
ingreso = int(input('Ingreso (en pesos): '))
nacimiento = int(input('Año de nacimiento: '))
hijos = int(input('Número de hijos: '))
pertenencia = int(input('Años de pertenencia al banco: '))
es_civil = input('Estado civil("S":soltero, "C":casado): ')
vivienda = input('Si vive en campo o ciudad ("U":urbano, "R":rural): ')

edad = 2021 - nacimiento

if pertenencia > 10 and hijos >= 2:
    print('APROBADO')

elif es_civil == 'C' and hijos > 3 and (edad >= 45 and edad <= 55):
    print('APROBADO')

elif ingreso > 2500000 and es_civil == 'S' and vivienda == 'U':
    print('APROBADO')

elif ingreso > 3500000 and pertenencia > 5:
    print('APROBADO')

elif vivienda == 'R' and es_civil == 'C' and hijos < 2:
    print('APROBADO')

else:
    print('RECHAZADO')     