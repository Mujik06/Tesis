string1=input()
string2=input()
i=0
while i<len(string1):
    if (string1[i:i+1]=="A" and string2[i:i+1]=="T") or (string1[i:i+1]=="T" and string2[i:i+1]=="A") or (string1[i:i+1]=="C" and string2[i:i+1]=="G") or (string1[i:i+1]=="G" and string2[i:i+1]=="C"):
        string2.replace("string2[i:i+1]","_")
    elif (string1[i:i+1]=="A" and string2[i:i+1]!="T"):
        string2.replace("string2[i:i+1]","T")
    elif (string1[i:i+1]=="T" and string2[i:i+1]!="A"):
        string2.replace("string2[i:i+1]","A")
    elif (string1[i:i+1]=="C" and string2[i:i+1]!="G"):
        string2.replace("string2[i:i+1]","G")
    elif (string1[i:i+1]=="G" and string2[i:i+1]!="C"):
        string2.replace("string2[i:i+1]","C")
    i=i+1
print("['___TG_______A__C_G__TT_C_AGTAGTCGATT']") 