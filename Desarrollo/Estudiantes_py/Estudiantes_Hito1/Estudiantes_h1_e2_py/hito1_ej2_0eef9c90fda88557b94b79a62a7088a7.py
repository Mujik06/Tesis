#Contestador de celular
numero = int(input("Ingrese numero telefonico: "))
hora = int(input("Ingrese hora de la llamada: "))
contestar = False
if len(str(numero)) != 8:
    print("Ingrese un numero telefonico de 8 cifras. ")
if hora < 0 or hora > 23:
    print("Porfavor ingrese una hora valida.")
elif 0 <= hora <= 7:
    contestar = True
elif hora < 14:
    contestar = False
    if (numero%1000) == 909:
        contestar = True
elif 17 <= hora <= 19:
    contestar = True
    if (numero//100000) == 877:
        contestar = False
elif hora > 19 and hora <= 23:
    contestar = False
else:
    contestar = False

if contestar == True:
    print("Resultado: CONTESTAR")
else:
    print("Resultado: NO CONTESTAR")