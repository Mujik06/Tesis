#Contestador de celular
NDT = int(input("Coloque un numero de telefono: "))
HDL = int(input("Coloque la hora de llamada: "))

while True:
    if HDL>=0 and HDL<=7:
        print("CONTESTAR") 
        break
    if HDL<14:
        if NDT%1000==909:
            print("CONTESTAR")           
            break
    if HDL>=17 and HDL<=19:
        if NDT//1000==877:
            print("CONTESTAR")            
            break
    if HDL>19:
        print("NO CONTESTAR")
        break
    else:
        print("NO CONTESTAR")        
        break