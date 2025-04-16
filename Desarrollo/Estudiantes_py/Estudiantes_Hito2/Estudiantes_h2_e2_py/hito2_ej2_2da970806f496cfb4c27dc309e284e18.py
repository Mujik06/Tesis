def sec_valida(adn):    
    adn=adn.upper()
    a="secuencia correcta"
    for i in adn:
        if i !="A":
            if i !="C":
                if i !="T":
                    if i !="G":
                        a="secuencia incorrecta"
    return a
if __name__ == "__main__":
    adn=input("Ingrese la secuencia:")
    print(sec_valida(adn))
    
     