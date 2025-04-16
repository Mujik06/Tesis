#Contestador de celular
# Entradas

n_telefono = int(input("Ingresa un numero de telefono: "))
hora = int(input("Ingresa una hora de llamada: "))

# Calculo

ultimas_tres_cifras = n_telefono % 1000
primeras_tres_cifras = n_telefono //100000

if 0 <= hora <= 7:
    print("Contestar la llamada")
    
elif hora <= 14 and ultimas_tres_cifras == 909:
    print("Contestar la llamada")

elif 17 <= hora <= 19 and primeras_tres_cifras != 877:
    print("Contestar la llamada")
    
elif hora > 19:
    print("No contestar la llamada")

else:
    print("No contestar la llamada")