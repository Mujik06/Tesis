def secuencia_adn(txt1,txt2):

        lista1=list(txt1)
        lista2=list(txt2)
        
        resultado=[]
        
        largo_lista1=len(lista1)
        largo_lista2=len(lista2)
        j=0
        fin="N"
        for i in range(largo_lista2):
            c=lista2[i]

            while(j<largo_lista1):
                s=lista1[j]
                if(c==s):
                    resultado.append(c)
                    j=j+1
                    if(j==largo_lista1):
                        fin="S"
                    break
                else:
                    resultado.append("_")
                j=j+1

            if(j==largo_lista1):
                if(fin=="S"):
                    fin="N"
                else:    
                    resultado.append(c)
        
        return("".join(resultado))

        
a=input("texto1: ")
b=input("texto2: ")
print(secuencia_adn(a,b))