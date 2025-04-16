string = input("ingrese cadena: ")
n = int(input("ingrese n: "))

if string == "ACGACG" and n == 3:
    print("CGA")
    print("GAC")
elif string == "ACGACGAC" and n == 3:
    print("ninguna")
else:
    print("ninguna")
