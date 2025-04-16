ingreso = int(input("colocar su ingreso: "))
ADN = int(input("colocar el año de nacimiento: "))
NDH = int(input("colocar el numero de hijos: "))
APB = int(input("colocar el numero de años que llevas en el banco: "))
EC = input("coloque su estado civil, (S) soltero, (C) casado: ")
CC = input("coloque donde vive, (U) urbano, (R) rural: ")

C = "C"
S = "S"
U = "U"
R = "R"

if (APB>10 and NDH>=2) or (EC==C and NDH>3 and (ADN>=1966 and ADN<=1976)) or (ingreso>2500000 and EC==S and CC==U) or (ingreso>3500000 and APB>5) or (CC==R and EC==C and NDH<2):    
    print("APROBADO")    
else:
    print("RECHAZADO")
    
        