pregunta_numero = input('Ingrese su número de celular: ')
pregunta_horario = int(input('Ingresar sólo hora del día: '))
numero_excepcional = pregunta_numero[5:9]

# PROCESAMIENTO: _________________________________________________________

if pregunta_horario == 0 or pregunta_horario <= 7 and numero_excepcional == '909':
    print('CONTESTANDO: llamada en el rango 00:00 a 07:00 horas')
    print('CONTESTAR')
elif pregunta_horario <= 14 and numero_excepcional == '909':
    print('CONTESTANDO: llamada en el rango 14:00 horas a número excepcional finalizado en 909')
    print('CONTESTAR')
elif pregunta_horario >= 17 and pregunta_horario <= 19 and numero_excepcional == '877':
    print('CONTESTANDO: llamada en el rango 17:00 a 19:00 horas finalizado 877')
    print('CONTESTAR')
else:
    print('NO DISPONIBLE: Me puedes llamar entre 00 a 07 horas. Gracias')
    print('NO CONTESTAR')
# SALIDA: ________________________________________________________________