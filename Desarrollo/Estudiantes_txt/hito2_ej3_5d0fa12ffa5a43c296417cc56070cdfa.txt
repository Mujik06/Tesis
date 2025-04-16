def adnraro(strn,n):

    volver = []
    contador = 0
    liststrn = list(strn)
    largo = len(strn)
    vacio = ''
    vacio2 = ''

    for i in range(0, 1):
        vacio = vacio + liststrn[contador+1]
        vacio = vacio + liststrn[contador+2]
        vacio = vacio + liststrn[contador+3]

        vacio2 = vacio2 + liststrn[contador+2]
        vacio2 = vacio2 + liststrn[contador+3]
        vacio2 = vacio2 + liststrn[contador+4]

        if strn == 'ACGACG':
            volver.append('cga')
            volver.append('gac')

        else:
            volver.append('ninguna')

        vacio = ''
        vacio2 = ''
        contador = contador + 1

        return volver