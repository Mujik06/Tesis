adn=str(input("Ingrese secuencia: ")).upper()
def secuencia(adn):
    for i in adn:
        if i=="A":
            pass
        elif i=="C":
            pass
        elif i=="G":
            pass
        elif i=="T":
            pass
        else:
            return True

if not secuencia(adn):
    print("La secuencia",str(adn),"es correcta")
else:
    print("La secuencia",str(adn),"es incorrecta")