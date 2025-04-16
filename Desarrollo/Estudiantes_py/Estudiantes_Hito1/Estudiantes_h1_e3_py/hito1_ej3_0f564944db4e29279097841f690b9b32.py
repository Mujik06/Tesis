#Aprobación de créditos
ingreso=int(input('Ingreso(en pesos):'))
año=int(input('Edad:'))
hijos=int(input('Numero de hijos:'))
AñosenBanco=int(input('Años de pertenencia al banco:'))
EstadoCivil=input('S:soltero, C:casado:')
lugar=input('U:urbano, R:rural:')
EstadoCivil=EstadoCivil.upper()
lugar=lugar.upper()

if AñosenBanco>=10 and hijos>=2:
    print('APROBADO')
elif EstadoCivil=="C" and hijos>3 and año>=45 and año<=55:
    print('APROBADO')
elif ingreso>2500000 and EstadoCivil=="S" and lugar=="U" and año>=0 and hijos>=0 and AñosenBanco>=0:
    print('APROBADO')
elif ingreso>3500000 and AñosenBanco>5:
    print('APROBADO')
elif lugar=="R" and EstadoCivil=="C" and hijos<2:
    print('APROBADO')
else:
    print('RECHAZADO')      