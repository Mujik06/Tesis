#Contestador de celular
numero_telefono= int(input("Ingrese un numero de telefono de 8 cifras: "))
hora= eval(input("Cuál es la hora del dia?: "))
final= numero_telefono % 1000
inicio= numero_telefono // 100000
if hora>=0 and hora <=7:
    print("CONTESTAR")
else:
    if hora>7 and hora<14:
        if final == 909:
            print("CONTESTAR")
        else:
            print("NO CONTESTAR")
    else:
        if 19>=hora>=17:
            if inicio==877:
                print("NO CONTESTAR")
            else:
                print("CONTESTAR")
        else:
            if 19<hora<23:
             print("NO CONTESTAR")