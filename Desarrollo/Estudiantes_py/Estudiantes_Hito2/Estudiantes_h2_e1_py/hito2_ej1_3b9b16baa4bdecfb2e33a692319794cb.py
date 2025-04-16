def alineamiento_ADN(ADN1,ADN2):

    i=0
    j=0
    ultimacadena=""

    while True:

        x=ADN1[i]
        y=ADN2[j]

        if x==y:
            ultimacadena=ultimacadena+(ADN2[j])
            i=i+1
            j=j+1

        elif x!=y:
            ultimacadena=ultimacadena+("_")
            i=i+1            

        if i==len(ADN1)-1:
            ultimacadena=ultimacadena+(ADN2[j:len(ADN2)])
            break

    return(ultimacadena)

ADN1=input()
ADN2=input()

print(alineamiento_ADN(ADN1,ADN2))