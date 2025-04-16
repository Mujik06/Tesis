#Contestador de celular
fono=str(input("Ingresa el número de teléfono (8 cifras) "))
hora=int(input("Ingresa la hora de la llamada "))
if(hora>=0 and hora<=7):
    print("CONTESTAR")
elif(hora<14):
    if(fono[5:]=="909"):
        print("CONTESTAR")
elif(hora>17 and hora<19):
    if(fono[5:]=="877"):
        print("CONTESTAR")
    else:
        print("NO CONTESTAR")
else:
    print("NO CONTESTAR")     