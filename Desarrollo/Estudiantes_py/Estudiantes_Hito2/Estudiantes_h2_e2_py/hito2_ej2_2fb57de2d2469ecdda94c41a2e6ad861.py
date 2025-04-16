adn1=input("Inserte secuencia: ").upper()
a=adn1.find("A")
t=adn1.find("T")
g=adn1.find("G")
c=adn1.find("C")
if a==-1 or t==-1 or g==-1 or c==-1 or adn1.find("B")!=-1:
    print("secuencia incorrecta")
if a!=-1 and t!=-1 and g!=-1 and c!=-1 and adn1.find("B")==-1:
    print("secuencia correcta")

