def alineamiento_ADN(ADN1,ADN2):

    i=0
    j=0
    cadena_final=""

    while True:

        x=ADN1[i]
        y=ADN2[j]

        if x==y:
            cadena_final=cadena_final+(ADN2[j])
            i=i+1
            j=j+1

        elif x!=y:
            cadena_final=cadena_final+("_")
            i=i+1            

        if i==len(ADN1)-1:
            cadena_final=cadena_final+(ADN2[j:len(ADN2)])
            break

    return(cadena_final)

ADN1=input()
ADN2=input()

print(alineamiento_ADN(ADN1,ADN2))