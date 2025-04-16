def alineamiento(s1,s2):
    let=""
    for i in s1:
        for k in s2:
            if i == k:
                 let+= k
        else:
            let+= '__'
    return let      

s1 = ('ACCTGGTTCTGTAGTCAGGATTACTA')
s2 = ('TGACGTTCAGTAGTCGATT')
print(alineamiento(s1,s2))