"""
Análisis Visual de Diversity Sampling
Muestra cómo K-Means agrupa las imágenes en clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys

# Importar funciones desde AL_functions.py
sys.path.insert(0, '../..')
from AL_functions import diversity_sampling, get_feature_extractor
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


# ========================================
# CONFIGURACIÓN
# ========================================

NUM_CLASSES = 3
SELECTED_CLASSES = [0, 1, 8]  # airplane, automobile, ship
CLASS_NAMES = ['Airplane', 'Automobile', 'Ship']

LABELED_SIZE = 500  # Primeras 500 para entrenar
START_INDEX = 501   # Empezar análisis desde 501
END_INDEX = 1000     # Hasta 600 (100 imágenes)

N_CLUSTERS = 20     # Número de clusters para K-Means

MODEL_PATH = '../../models/500_train/best_model.keras'


# ========================================
# 1. CARGAR DATOS Y MODELO
# ========================================

def load_data():
    """Carga CIFAR-10 filtrado"""
    
    print(" Cargando CIFAR-10...")
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    
    # Filtrar clases
    mask = np.isin(y_train.flatten(), SELECTED_CLASSES)
    x_train = x_train[mask]
    y_train = y_train[mask]
    
    # Remapear labels
    for new_idx, old_idx in enumerate(SELECTED_CLASSES):
        y_train[y_train == old_idx] = new_idx
    
    # Normalizar
    x_train = x_train.astype('float32') / 255.0
    
    print(f" Dataset: {len(x_train)} imágenes")
    return x_train, y_train


def load_model():
    """Carga el modelo entrenado"""
    sys.path.insert(0, '..')
    print(f" Cargando modelo: {MODEL_PATH}")
    
    if not Path(MODEL_PATH).exists():
        print(f" ERROR: No se encuentra el modelo en {MODEL_PATH}")
        exit(1)
    
    model = keras.models.load_model(MODEL_PATH)
    print(" Modelo cargado")
    return model


# ========================================
# 3. APLICAR DIVERSITY SAMPLING
# ========================================

def analyze_diversity(model, images_new, n_clusters=20):
    """
    Aplica diversity_sampling y analiza los resultados
    
    Returns:
        selected_indices: índices de las seleccionadas (representantes)
        cluster_labels: cluster asignado a cada imagen
        distances_to_centroid: distancia de cada imagen a su centroide
    """
    
    print(f" Aplicando Diversity Sampling (K={n_clusters})...")
    
    # Usar diversity_sampling de AL_functions.py
    selected_indices, diversity_scores = diversity_sampling(
        model, images_new, n_samples=n_clusters
    )
    
    print(f" Diversity sampling completado")
    print(f"  Seleccionadas: {len(selected_indices)} representantes (1 por cluster)")
    
    # Para análisis adicional, extraer features y asignar clusters manualmente
    feature_extractor = get_feature_extractor(model)
    features = feature_extractor.predict(images_new, verbose=0, batch_size=32)
    
    # K-Means para obtener cluster_labels y distancias
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Calcular distancias de cada imagen a su centroide
    distances_to_centroid = []
    for i, label in enumerate(cluster_labels):
        centroid = kmeans.cluster_centers_[label]
        dist = euclidean_distances(
            features[i].reshape(1, -1),
            centroid.reshape(1, -1)
        )[0][0]
        distances_to_centroid.append(dist)
    
    distances_to_centroid = np.array(distances_to_centroid)
    
    return selected_indices, cluster_labels, distances_to_centroid


# ========================================
# 4. VISUALIZACIÓN
# ========================================

def plot_diversity_analysis(images, labels, selected_indices, 
                            cluster_labels, distances, start_idx):
    """
    Crea 5 gráficos, cada uno con 20 imágenes
    """
    
    print(f" Creando gráficos...")
    
    Path('results').mkdir(exist_ok=True)
    
    # Crear máscara booleana de seleccionadas
    is_selected = np.zeros(len(images), dtype=bool)
    is_selected[selected_indices] = True
    
    # Calcular umbral de distancia (percentil 75)
    threshold = np.percentile(distances, 75)
    is_far = distances > threshold
    
    # 5 gráficos de 20 imágenes cada uno
    for graph_num in range(5):
        
        fig, axes = plt.subplots(4, 5, figsize=(18, 14))
        fig.suptitle(f'Diversity Sampling Analysis (K-Means) - Gráfico {graph_num+1}/5\n'
                    f'Imágenes {start_idx + graph_num*20} - {start_idx + (graph_num+1)*20}',
                    fontsize=16, fontweight='bold')
        
        # 20 imágenes por gráfico
        for i in range(20):
            idx = graph_num * 20 + i
            
            if idx >= len(images):
                break
            
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            # Mostrar imagen
            ax.imshow(images[idx])
            ax.axis('off')
            
            # Información
            true_label = int(labels[idx])
            cluster_id = cluster_labels[idx]
            distance = distances[idx]
            selected = is_selected[idx]
            is_novel = is_far[idx]
            
            # Título con info
            title = f'Real: {CLASS_NAMES[true_label]}\n'
            title += f'Cluster: {cluster_id}\n'
            title += f'Distancia: {distance:.3f}\n'
            
            # Estado
            if selected:
                title += 'CLUSTER CENTROID'
                color = 'blue'
                edge_color = 'blue'
                edge_width = 3
            elif is_novel:
                title += 'FAR FROM CENTROID'
                color = 'red'
                edge_color = 'red'
                edge_width = 2
            else:
                title += 'NEAR CENTROID'
                color = 'green'
                edge_color = 'green'
                edge_width = 1
            
            ax.set_title(title, fontsize=8, color=color, fontweight='bold')
            
            # Borde
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(edge_width)
        
        plt.tight_layout()
        filename = f'results/diversity_analysis_{graph_num+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Guardado: {filename}")
        plt.close()


# ========================================
# 5. ESTADÍSTICAS
# ========================================

def print_statistics(selected_indices, cluster_labels, distances, labels):
    """Imprime estadísticas del clustering"""
    
    print()
    print("="*60)
    print(" ESTADÍSTICAS DE DIVERSITY")
    print("="*60)
    
    # Distribución por cluster
    print("\nDistribución por cluster:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id:2d}: {count:2d} imágenes")
    
    # Representantes seleccionados
    print(f"\nRepresentantes seleccionados: {len(selected_indices)}")
    
    # Umbral
    threshold = np.percentile(distances, 75)
    is_far = distances > threshold
    
    # Imágenes lejanas por clase real
    print(f"\nImágenes LEJANAS del centroide (top 25%) por clase:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = (labels == class_id) & is_far
        count = mask.sum()
        total = (labels == class_id).sum()
        print(f"  {class_name:12s}: {count:2d}/{total:2d} ({count/total*100:.1f}%)")
    
    print(f"\nTotal imágenes LEJANAS: {is_far.sum()}/{END_INDEX-START_INDEX} ({is_far.sum() / (END_INDEX-START_INDEX) * 100:.1f}%)")
    print(f"Umbral usado: {threshold:.3f} (percentil 75)")


# ========================================
# 6. MAIN
# ========================================

def main():
    
    print("="*60)
    print(" ANÁLISIS DE DIVERSITY SAMPLING")
    print("="*60)
    print()
    
    # Cargar datos y modelo
    x_train, y_train = load_data()
    model = load_model()
    
    print()
    print(f" Configuración:")
    print(f"  - Analizar: imágenes {START_INDEX}-{END_INDEX} (100 imágenes)")
    print(f"  - K-Means clusters: {N_CLUSTERS}")
    print()
    
    # Aplicar diversity sampling usando AL_functions.py
    selected_indices, cluster_labels, distances = analyze_diversity(
        model, x_train[START_INDEX:END_INDEX], N_CLUSTERS
    )
    
    # Crear gráficos
    print()
    plot_diversity_analysis(
        x_train[START_INDEX:END_INDEX],
        y_train[START_INDEX:END_INDEX].flatten(),
        selected_indices,
        cluster_labels,
        distances,
        START_INDEX
    )
    
    # Estadísticas
    print_statistics(
        selected_indices,
        cluster_labels, 
        distances,
        y_train[START_INDEX:END_INDEX].flatten()
    )
    
    print()
    print(" Análisis completado!")
    print(f" Gráficos guardados en: results/diversity_analysis_*.png")

    threshold = np.percentile(distances, 75)
    is_far = distances > threshold
    np.save('results/diversity_indices.npy', np.where(is_far)[0])
    print("💾 Índices guardados: results/diversity_indices.npy")


if __name__ == "__main__":
    main()