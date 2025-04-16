string = str(input('ingresa la secuencia: '))
entero = int(input('ingresa n: '))

def conversor(string, entero):
  confirmador = len(string) % entero
  contador = len(string) / entero
  #return type(confirmador)
  if (confirmador) == 0:
    string_2 = string[1]+string[2]+string[0]
    string_3 = string[-1]+string[-3]+string[-2]
    return print(string_2, string_3)
  else:
    print('ninguna')
conversor(string, entero)