ADN = ["A", "a", "G", "g", "C", "c", "T", "t"]
def verificarSecuencia(string):
    palabra = ""
    for letras in string:
        if letras not in ADN:
            return "secuencia incorrecta"
        else:
            palabra += letras
    if palabra == string:
        return "secuencia correcta"

adn = input("ingrese la secuencia: ")
verificado = verificarSecuencia(adn)
print("La secuencia", adn, verificado)
