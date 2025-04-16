#Contestador de celular
numero = str(input("Ingrese el numero telefonico: "))
hora = int(input("Ingrese la hora de la llamada: "))
var_1 = numero[5:9]
var_2 = numero[0:3]
int(var_1)
int(var_2)
if hora >= 0 and hora <= 19:
    if hora <= 7:
        print("CONTESTAR")
    elif hora >= 7 and hora <= 14:
        if var_1 == "909":
            print("CONTESTAR")
        else:
            print("NO CONTESTAR")
    elif hora >= 17 and hora <= 19:
        if(var_2 == "877"):
            print("NO CONTESTAR")
        else:
            print("CONTESTAR")
else:
   print("NO CONTESTAR")