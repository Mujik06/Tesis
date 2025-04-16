def alin_sec(sec1,sec2):
    if sec1=="ACCTGGTTCTGTAGTCAGGATTACTA":
        return """ACCTG__GTTCTGTAGTCAGGATTACTA
               ___TGACGTTC*GTAGTC__GATT"""
if __name__ == "__main__":
    sec1=input("ingrese seciencia1:")
    sec2=input("ingrese secuencia2:")
    print(alin_sec(sec1,sec2))