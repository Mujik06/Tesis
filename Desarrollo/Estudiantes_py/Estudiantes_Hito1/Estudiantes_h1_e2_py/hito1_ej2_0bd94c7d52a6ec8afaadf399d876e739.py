#Contestador de celular
#Entradas
NT = int(input("Ingrese número de telefono: "))
HL = int(input("Ingrese hora de la llamada: "))
if 0<= HL <= 7:
    print("CONTESTAR")

elif 8<= HL <= 14:
    ver = NT%1000
    if ver == 909:
        print("CONTESTAR")
    else:
         print("NO CONTESTAR")

elif 15 <= HL <= 16:
     print("NO CONTESTAR")

elif 17 <= HL <= 19:
    ver = NT//100000
    if ver == 877:
         print("NO CONTESTAR")
    else:
         print("CONTESTAR")

else:
     print("NO CONTESTAR")