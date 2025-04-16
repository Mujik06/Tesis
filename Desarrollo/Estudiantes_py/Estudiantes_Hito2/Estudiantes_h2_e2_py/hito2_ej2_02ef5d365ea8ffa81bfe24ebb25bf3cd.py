def chequeo_adn(string1):
    string1 = string1.lower()
    has_different_letter = False
    for letter in string1:
        if letter != 'a' and letter != 't' and letter != 'c' and letter != 'g':
            has_different_letter = True
    if not has_different_letter:
        print('secuencia correcta')
    else:
        print('secuencia incorrecta')

chequeo_adn(input())
