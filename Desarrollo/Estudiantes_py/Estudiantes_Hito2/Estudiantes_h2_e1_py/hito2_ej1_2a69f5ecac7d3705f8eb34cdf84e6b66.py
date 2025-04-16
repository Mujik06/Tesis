def alinear_secuencias(string1,string2):
    modificada=""
    j=0
    carateres=0
    modificada=""
    for i in range(len(string2)):
        while j<len(string1):
            if string2[i]==string1[j]:
                modificada+=string2[i]
                carateres+=1
                j=j+1
                break
            else:
                modificada+="_"
            j+=1
        if len(modificada)==len(string1):
            modificada+=string2[carateres:]
    return modificada

#cadena1="ACCTGGTTCTGTAGTCAGGATTACTA"
#cadena2="TGACGTTCAGTAGTCGATT"

cadena1=input("Cadena 1: ")
cadena2=input("Cadena 2: ")
            
print(alinear_secuencias(cadena1,cadena2))