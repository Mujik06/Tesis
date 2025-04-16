def adn(string):
    genoma=string.upper()
    i=0
    suma=0
    while i<len(genoma):
        if genoma[i]!="A" and genoma[i]!="C" and genoma[i]!="T" and genoma[i]!="G":
            suma=suma+1
            i=i+1
        else:
            i=i+1
    if suma>0:
        return "secuencia incorrecta"
    else:
        return "secuencia correcta"
a=input("adn")
print(adn(a))
