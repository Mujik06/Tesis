#Contestador de celular
telefono = "0"
hora = 24

while(len(telefono) != 8):
    telefono = input("Ingrese numero telefonico de 8 digitos: ")

while(hora < 0 or hora > 23):
    hora = int(input("Ingrese hora de la llamada: "))

if (hora >= 0 and hora <= 7):
    print("Resultado: CONTESTAR")

elif (hora <= 14):
    if (telefono[5:] == "909"):
        print("Resultado: CONTESTAR")
    else:
        print("Resultado: NO CONTESTAR")

elif (hora < 17):
    print("Resultado: NO CONTESTAR")

elif (hora <= 19):
    if (telefono[0:2] == "87"):
        print("Resultado: NO CONTESTAR")
    else:
        print("Resultado: CONTESTAR")

else:
    print("Resultado: NO CONTESTAR")