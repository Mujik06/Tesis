#Contestador de celular
C=int(input("Ingrese celular: "))
if C/10000000>=1 and C/10000000<10:
    H= int(input("Ingrese hora:"))
    if H>=0 and H<=23:
        if H>0 and H<7:
            print("Contestar")
        elif H<14 and (C-909)%1000==0:
            print("Contestar")
        elif H>5 and H<7 and (C-877)%1000!=0:
            print("Contestar")
        else:
            print("No contestar")


    