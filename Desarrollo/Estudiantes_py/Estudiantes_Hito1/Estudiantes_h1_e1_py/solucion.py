#Nota Final

pt = float(input('Ingrese PT: '))
pi = float(input('Ingrese PI: '))
ne = float(input('Ingrese NE: '))
pp = float(input('Ingrese PP: '))

promedio = 0.3*pt + 0.3*pi + 0.3*ne + 0.1*pp
redondeo = round(promedio,1)
print(str(redondeo))
print(str(promedio)) #para comprobar efectivamente el redondeo
