us=input("Ingrese su cadena: ")
oneexist=False
l = []
x = 0
r = ""
while True:
    try:
        y=int(float(input("Ingrese el largo: ")))
        if (y<=1):
            assert y>=2
        elif(y>len(us)):
            assert y<len(us)
        break
    except AssertionError:
        if (y<=1):
            print("Debe ser mayor que uno")
        elif (y > len(us)):
            print("Debe ser menor que la longitud de la cadena, la cual es: ",len(us))
    except ValueError:
        print("Error, debe ser numerico")
while y<=len(us):
    for a in range(x,y):
        r+=us[a]
    l.append(r)
    r=""
    x+=1
    y+=1
for b in l:
    if (l.count(b)==1):
        print(b)
        if oneexist==False:
            oneexist=True
if not oneexist:
    print("Ninguna")