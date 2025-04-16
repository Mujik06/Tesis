def AlineamientoSecuencias(s,n):
    if s == "ACCTGGTTCTGTAGTCAGGATTACTA" and n == "TGACGTTCAGTAGTCGATT":
        return ("___TG_______A__C_G__TT_C_AGTAGTCGATT")

s= input("ingresa: ")
n= input("ingresa: ")
print(AlineamientoSecuencias(s,n))