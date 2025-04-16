adn1=input()
adn2=input()
i=0
j=0
adn_final=""
while True:
    x=adn1[i]
    y=adn2[j]
    if x==y:
        adn_final=adn_final+(adn2[j])
        i=i+1
        j=j+1
    elif x!=y:
        adn_final=adn_final+("_")
        i=i+1            
    if i==len(adn1)-1:
        adn_final=adn_final+(adn2[j:len(adn2)])
        break

print(adn_final)
