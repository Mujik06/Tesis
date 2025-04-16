#Contestador de celular
n=input("ingrese su número de celular: ")
seguir=True
while seguir:
    if len(n)==8:
        hora=input("ingrese la hora de la llamada (solo los dos primeros digitos): ")
        if hora<="07":
            print("contestar")
            break
        elif hora=="14":
            if int(n[6])==9 and int(n[7])==0 and int(n[8])==9:
                print("contestar")
                break
            else:
                print("no contestar")
                break
        elif "17"<=hora<="19":
            if int(n[1])==8 and int(n[2])==7 and int(n[3])==7:
                print("contestar")
                break
            else:
                print("no contestar")
                break
        elif hora>"19":
            print("no contestar")
            break
        else:
            print("contestar")
            break
    else:
        n = input("vuelva a ingresar el número")    