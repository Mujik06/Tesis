adn = []
adn.append(input("Ingrese el adn: "))
if adn != "ACTG":
    print("La secuencia ",adn, " es incorrecta")
if adn in ["ACTG"]:
    print("La secuencia es correcta")