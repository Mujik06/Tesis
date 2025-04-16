#Contestador de celular
print("ingese su numero de telefono (8 cifras) y la hora")

numero = int(input("ingrese su numero de telefeono: "))
hora = int(input("ingrese la hora de llamada: "))

if (hora <= 7):
    print("numero telefonico: ", numero)
    print("hora: ", hora)
    print("resultado: CONTESTAR")

elif hora>7 and hora<=14:
    print("numero telefonico: ", numero)
    print("hora: ", hora)
    print("resultado: CONTESTAR")

elif hora>=17 and hora<=19:
    print("numero telefonico: ", numero)
    print("hora: ", hora)
    print("resultado: no CONTESTAR")

elif hora>19:
    print("numero telefonico: ", numero)
    print("hora: ", hora)
    print("resultado: NO CONTESTAR")
