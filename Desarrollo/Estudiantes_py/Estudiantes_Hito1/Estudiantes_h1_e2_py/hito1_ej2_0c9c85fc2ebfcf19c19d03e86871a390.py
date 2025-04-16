#Contestador de celular
nrotlf = int(input("ingrese numero telefonico: "))

comienzonro= nrotlf // 100000
finalnro= nrotlf % 1000


horallamada = int(input("Ingrese hora de la llamada: "))

if ((horallamada > 23) or (horallamada < 8) ):
    print("CONTESTAR")

elif (horallamada < 14) and (horallamada >7) and finalnro == 909:
    print("CONTESTAR")

elif (horallamada < 14) and (horallamada >7):
    print("NO CONTESTAR")

elif (horallamada >16) and (horallamada < 20) and comienzonro == 877:
    print("NO CONTESTAR")

elif (horallamada >16) and  (horallamada < 20) :
    print("CONTESTAR0")

elif (horallamada >18) and  (horallamada < 24) :
    print("NO CONTESTAR")