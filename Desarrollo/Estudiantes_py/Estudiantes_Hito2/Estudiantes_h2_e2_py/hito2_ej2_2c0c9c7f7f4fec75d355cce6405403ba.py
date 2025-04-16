adn=input()
i=0
while i<len(adn):
  if adn[i]!="a" or adn[i]!="e" or adn[i]!="i" or adn[i]!="o" or adn[i]!="u":
    print("secuencia incorrecta")
    i+=1
  else:
    print("secuencia correcta")