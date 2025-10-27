import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys

sys.path.insert(0, '../..')
from AL_functions import diversity_sampling, get_feature_extractor
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


NUM_CLASSES = 3
SELECTED_CLASSES = [0, 1, 8]
CLASS_NAMES = ['Airplane', 'Automobile', 'Ship']
MODEL_PATH = '../../models/500_train/best_model.keras'


def load_cifar10_filtered():
    """Carga CIFAR-10 filtrado por clases seleccionadas"""
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    
    mask = np.isin(y_train.flatten(), SELECTED_CLASSES)
    x_train = x_train[mask]
    y_train = y_train[mask]
    
    for new_idx, old_idx in enumerate(SELECTED_CLASSES):
        y_train[y_train == old_idx] = new_idx
    
    x_train = x_train.astype('float32') / 255.0
    
    return x_train, y_train.flatten()


def calculate_diversity(model, images, n_clusters=20):
    """
    Aplica diversity sampling y calcula clustering
    
    Returns:
        selected_indices: Índices de representantes (1 por cluster)
        cluster_labels: Cluster asignado a cada imagen
        distances_to_centroid: Distancia de cada imagen a su centroide
    """
    selected_indices, _ = diversity_sampling(model, images, n_samples=n_clusters)
    
    feature_extractor = get_feature_extractor(model)
    features = feature_extractor.predict(images, verbose=0, batch_size=128)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    distances_to_centroid = np.array([
        euclidean_distances(
            features[i].reshape(1, -1),
            kmeans.cluster_centers_[cluster_labels[i]].reshape(1, -1)
        )[0][0]
        for i in range(len(images))
    ])
    
    return selected_indices, cluster_labels, distances_to_centroid


def get_diverse_indices(distances, threshold_percentile=75):
    """Retorna índices de imágenes lejos de centroides"""
    threshold = np.percentile(distances, threshold_percentile)
    is_far = distances > threshold
    return np.where(is_far)[0], is_far


def visualize_diversity(images, labels, selected_indices, cluster_labels, 
                       distances, start_idx, output_dir='results'):
    """Genera visualizaciones de diversity analysis"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    is_selected = np.zeros(len(images), dtype=bool)
    is_selected[selected_indices] = True
    
    threshold = np.percentile(distances, 75)
    is_far = distances > threshold
    
    num_images = len(images)
    num_graphs = (num_images + 19) // 20
    
    for graph_num in range(num_graphs):
        fig, axes = plt.subplots(4, 5, figsize=(18, 14))
        fig.suptitle(f'Diversity Sampling Analysis (K-Means) - Graph {graph_num+1}/{num_graphs}\n'
                    f'Images {start_idx + graph_num*20} - {start_idx + min((graph_num+1)*20, num_images)}',
                    fontsize=16, fontweight='bold')
        
        for i in range(20):
            idx = graph_num * 20 + i
            if idx >= num_images:
                break
            
            row, col = i // 5, i % 5
            ax = axes[row, col]
            
            ax.imshow(images[idx])
            ax.axis('off')
            
            true_label = int(labels[idx])
            cluster_id = cluster_labels[idx]
            distance = distances[idx]
            
            title = f'Real: {CLASS_NAMES[true_label]}\n'
            title += f'Cluster: {cluster_id}\n'
            title += f'Distance: {distance:.3f}\n'
            
            if is_selected[idx]:
                title += 'CENTROID'
                color, edge_color, edge_width = 'blue', 'blue', 3
            elif is_far[idx]:
                title += 'FAR'
                color, edge_color, edge_width = 'red', 'red', 2
            else:
                title += 'NEAR'
                color, edge_color, edge_width = 'green', 'green', 1
            
            ax.set_title(title, fontsize=8, color=color, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(edge_width)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/diversity_analysis_{graph_num+1}.png', dpi=150, bbox_inches='tight')
        plt.close()


def analyze_diversity(model_path, start_idx, end_idx, n_clusters=20, 
                      output_dir='results', verbose=True):
    """
    Análisis completo de diversity para un rango de imágenes
    
    Args:
        model_path: Ruta al modelo .keras
        start_idx: Índice inicial del rango a analizar
        end_idx: Índice final del rango a analizar
        n_clusters: Número de clusters para K-Means
        output_dir: Directorio para guardar resultados
        verbose: Mostrar mensajes de progreso
        
    Returns:
        diverse_indices: Índices de imágenes lejos de centroides
        clustering_info: Dict con cluster_labels, distances, selected_indices
    """
    if verbose:
        print(f"Loading data and model...")
    
    x_train, y_train = load_cifar10_filtered()
    model = keras.models.load_model(model_path)
    
    images = x_train[start_idx:end_idx]
    labels = y_train[start_idx:end_idx]
    
    if verbose:
        print(f"Analyzing {len(images)} images ({start_idx}-{end_idx})...")
        print(f"K-Means clusters: {n_clusters}")
    
    selected_indices, cluster_labels, distances = calculate_diversity(model, images, n_clusters)
    diverse_indices, is_far = get_diverse_indices(distances, threshold_percentile=75)
    
    if verbose:
        print(f"Found {len(diverse_indices)} diverse images ({len(diverse_indices)/len(images)*100:.1f}%)")
        print(f"  Representatives: {len(selected_indices)}")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"  Clusters: {len(unique)} with avg {counts.mean():.1f} images each")
    
    # visualize_diversity(images, labels, selected_indices, cluster_labels, distances, start_idx, output_dir)
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    np.save(f'{output_dir}/diversity_indices.npy', diverse_indices)
    
    if verbose:
        print(f"Results saved to {output_dir}/")
    
    return diverse_indices, {
        'cluster_labels': cluster_labels,
        'distances': distances,
        'selected_indices': selected_indices
    }


def main():
    """Ejecución standalone con parámetros por defecto"""
    LABELED_SIZE = 500
    START_INDEX = 501
    END_INDEX = 1000
    N_CLUSTERS = 20
    
    print("="*60)
    print("DIVERSITY SAMPLING ANALYSIS")
    print("="*60)
    print(f"Labeled dataset: first {LABELED_SIZE} images")
    print(f"Analyzing: images {START_INDEX}-{END_INDEX}")
    print(f"K-Means clusters: {N_CLUSTERS}")
    print()
    
    diverse_indices, info = analyze_diversity(
        model_path=MODEL_PATH,
        start_idx=START_INDEX,
        end_idx=END_INDEX,
        n_clusters=N_CLUSTERS,
        output_dir='results',
        verbose=True
    )
    
    print()
    print("="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Distances - Min: {info['distances'].min():.3f}, Max: {info['distances'].max():.3f}, Mean: {info['distances'].mean():.3f}")
    print(f"Threshold (75th percentile): {np.percentile(info['distances'], 75):.3f}")
    print()
    print("Analysis completed!")


if __name__ == "__main__":
    main()