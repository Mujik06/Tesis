pi = input("Ingrese su string: ")
pe = pi.split(",")
string = pe[0].lower()
n = int(pe[1])

if len(string)%n == 0:
    sigue = True
    listastr = []
    listaO = []
    p = "po"
    z = 0
    a = 0
    while sigue:
        linea = ""
        while a < n:
            linea += string[z]
            z+=1
            a+=1
        listastr.append(linea)
        if n == len(string):
            sigue = False
        n += n
    k = 0
    if k%2 == 0:
        lin = ""
        pu = []
        for i in listastr[k]:
            pu.append(i)
        lin = pu[1]+pu[2]+pu[0]
        listaO.append(lin)
        k+=1
    if k%2 != 0:
        lin = ""
        pu = []
        for i in listastr[k]:
            pu.append(i)
        lin = pu[1]+pu[2]+pu[0]
        listaO.append(lin)
        k+=1
        
    print(listaO)

else:
    print("ninguna")
    
          