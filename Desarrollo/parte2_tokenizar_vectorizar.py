from transformers import RobertaTokenizer, RobertaModel
from pathlib import Path
from collections import Counter
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Inicializar modelo y tokenizador de modelo pre entrenado
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

# Preparar ruta de datos .txt
ruta_carpeta = Path("C:/Users/joaqu/UNAB/Tesis/Desarrollo/Estudiantes_txt")
archivos_txt = list(ruta_carpeta.glob('*.txt'))

# Listas para almacenar Embeddings y Etiquedas de archivos de estudiantes
embeddings = []
etiquetas = []

# Lectura de carpeta con archivos .txt
if not archivos_txt:
    print("No se encontraron archivos .txt en la ruta especificada.")
else:
    for archivo_txt in archivos_txt:
        try:
            with open(archivo_txt, 'r', encoding='utf-8') as archivo:
                codigo = archivo.read()

            # Tokenizar codigo fuente
            tokens = tokenizer.tokenize(codigo)
            
            # Contar frecuencia de Tokens - Identificar los 2 mas esenciales
            frecuencia = Counter(tokens)
            tokens_esenciales = [tok for tok, _ in frecuencia.most_common(2)]

            # Condicional para codigos sin Tokens suficientes
            if not tokens_esenciales:
                print(f"No se encontraron tokens esenciales en {archivo_txt.name}")
                continue
            
            # Filtrar codigo Tokenizado y convertir los mas esenciales a IDs
            tokens_filtrados = [tok for tok in tokens if tok in tokens_esenciales]
            ids = tokenizer.convert_tokens_to_ids(tokens_filtrados)
            if not ids:
                print(f"No se pudieron generar IDs en {archivo_txt.name}")
                continue
            
            # Creacion de Tensores de entrada y Obtener Embeddings del modelo pre entrenado
            entrada = torch.tensor([ids])
            with torch.no_grad():
                salida = model(entrada)
                emb = salida.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(emb)
                etiquetas.append(archivo_txt.stem)

        except Exception as e:
            print(f"Error al procesar {archivo_txt.name}: {str(e)}")

    # Solo si hay datos válidos, verificar generacion de Embeddings
    if embeddings:
        embeddings = np.array(embeddings)

        # Metodo t-SNE
        if len(embeddings) > 1:
            tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings) - 1), random_state=0)
            emb_tsne = tsne.fit_transform(embeddings)
            
            # Crear etiquetas tipo "Estudiante 1", "Estudiante 2", ..., "solucion"
            etiquetas_visuales = []
            contador = 1
            for etq in etiquetas:
                if "solucion" in etq.lower():
                    etiquetas_visuales.append("Solución")
                else:
                    etiquetas_visuales.append(f"Estudiante {contador}")
                    contador += 1

            # Visualización de Grafico 
            plt.figure(figsize=(12, 8))
            for i, nombre in enumerate(etiquetas_visuales):
                plt.scatter(emb_tsne[i, 0], emb_tsne[i, 1])
                plt.annotate(nombre, (emb_tsne[i, 0], emb_tsne[i, 1]), fontsize=10)

            plt.title("t-SNE de Embeddings - Visualización por Estudiante")
            plt.xlabel("Componente 1")
            plt.ylabel("Componente 2")
            plt.grid(True)
            plt.show()

        # Metodo PCA
        if len(embeddings) > 1:
            pca = PCA(n_components=2)
            emb_pca = pca.fit_transform(embeddings)
        
        else:
            print("No hay suficientes vectores para aplicar PCA.")
    else:
        print("No se generaron embeddings válidos.")
