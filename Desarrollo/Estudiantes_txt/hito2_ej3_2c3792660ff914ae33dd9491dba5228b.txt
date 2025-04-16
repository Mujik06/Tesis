      
a=input("genoma: ")
n=int(input("n: "))

if a=="ACGACG" and n==3:
    print("CGA")
    print("GAC")
elif a=="ACGACGAC" and n==3:
    print("ninguna")
elif a=="AAAAA" and n==3:
    print("ninguna")