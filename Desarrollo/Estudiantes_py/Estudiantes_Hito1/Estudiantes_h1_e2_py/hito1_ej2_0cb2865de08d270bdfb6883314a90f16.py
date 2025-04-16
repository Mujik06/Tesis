#Contestador de celular
numero= int(input("Ingrese numero telefonico: "))
hora= int(input("Ingrese hora de la llamada: "))
if hora<=7:
    print("CONTESTAR")
if hora>=19:
   print("NO CONTESTAR")
if hora in range(17, 19) and numero % 100000000 // 100000 == 877:
    print("NO CONTESTAR")
if hora in range (7,14) and numero % 1000 !=909:
    print("NO CONTESTAR")
elif hora in range(8,14) and numero % 1000 ==909:
    print("CONTESTAR")