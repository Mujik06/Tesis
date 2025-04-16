#Aprobación de créditos
print("Bienvenido Al Banco")
pesos=int(input('Ingrese su ingreso en pesos:'))
year=int(input('Ingrese su año de nacimiento:'))
son=int(input('¿cuantos hijos tiene?:'))
by=int(input('¿cuantos años pertenece a este banco?:'))
estado=str(input('estado civil(s=soltero,c=casado):'))
vive=str(input('Localidad donde vive(u=urbano,r=rural):'))
s=int(1)
c=int(2)
u=int(1)
r=int(2)
#Aprovacion de Credito
if by>=10 and son>=2:
    print("APROBADO");
elif estado==2 and son>=3 and year<=1972 and year>=1962:
       print('APROBADO');
elif pesos>=2500000 and estado==1 and vive==1:
       print("APROBADO");
elif pesos>=3500000 and by>5:
       print('APROBADO');
elif vive!=1 and estado!=1 and son<=2:
    print('APROBADO');
else:
    print('RECHAZADO')
