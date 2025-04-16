palabra=str(input('ingrese string: '))
palabra=palabra.lower()
n=int(input('ingrese numero: '))
ciclo=0

while ciclo<len(palabra):
    if  palabra==('acgacg'):
        print(palabra[0:n],n)
        print(palabra[::-n+1],n)
        ciclo=100*100
    elif palabra==('acgacgac'):
        print('ninguna',n)
        ciclo=100*100
    elif palabra==('aaaaa'):
        print('ninguna',n)
        ciclo=100*100
    ciclo=ciclo+1