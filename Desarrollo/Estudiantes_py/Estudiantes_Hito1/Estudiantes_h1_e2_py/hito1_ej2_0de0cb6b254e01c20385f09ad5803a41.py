#Contestador de celular

numero = int(input("Ingrese el numero telefonico de ocho cifras: "))
hora = int(input("Ingrese la hora en que realizó la llamada: "))

if hora >= 0 and hora <= 7:
    print("CONTESTAR")

elif hora > 7 and hora <=14:
    if (numero%10**3) == 909:
        print("CONTESTAR")
    else:
        print("NO CONTESTAR")
        
elif hora >= 17 and hora <= 19:
    if(numero%10**3) == "877":
        print("CONTESTAR")
    else:
        print("NO CONTESTAR")

elif hora > 19 and hora <= 23:
    print("NO CONTESTAR")

else:
    print("NO CONTESTAR")