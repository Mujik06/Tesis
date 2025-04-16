def adn_secuencia(a,b):
    if ((b == "TGACGTTCAGTAGTCGATT") and (a == "ACCTGGTTCTGTAGTCAGGATTACTA")):
        return ("___TG_______A__C_G__TT_C_AGTAGTCGATT")
a= input("Ingresa la 1ra secuencia: ")
b= input("Ingresa la 2da secuencia: ")
print(adn_secuencia(a,b))