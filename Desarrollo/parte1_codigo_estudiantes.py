from pathlib import Path

# Definir rutas de origen y destino
ruta_py = Path("C:/Users/joaqu/UNAB/Tesis/Desarrollo/Estudiantes_py/Estudiantes_Hito2/Estudiantes_h2_e3_py")
ruta_txt = Path("C:/Users/joaqu/UNAB/Tesis/Desarrollo/Estudiantes_txt/")
ruta_txt.mkdir(parents=True, exist_ok=True)

# Leer cada archivo y convertirlos de .py a .txt
for archivo in ruta_py.glob("*.py"):
    contenido = archivo.read_text(encoding='utf-8')
    nuevo_txt = ruta_txt / (archivo.stem + ".txt")
    nuevo_txt.write_text(contenido, encoding='utf-8')
    print(f"Convertido: {archivo.name} -> {nuevo_txt.name}")
