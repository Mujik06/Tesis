nt = int(input("Escriba el numero telefonico"))
hora = int(input("Escriba la hora de la llamada"))
if hora >= 0 and hora <= 7:
    print("CONTESTAR")
else:
    if hora > 7 and hora < 14:
        r = (nt - 909)
        r1 = float(r/100)
        r2 = int(r/100)
        if r1==r2:
            print("CONTESTAR")
        else:
            if hora > 17 and hora < 19:
                r3 = nt - 87700000
                if r3 <=9999:
                    print("NO CONTESTAR")
                else:
                    print("CONTESTAR")
            else:
                if hora > 19:
                    print("NO CONTESTAR")
                else:
                    print("CONTESTAR")
    else:
        print("NO CONTESTAR")
                    
                    