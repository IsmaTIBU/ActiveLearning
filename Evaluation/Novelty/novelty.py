import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys

sys.path.insert(0, '../..')
from AL_functions import novelty_detection


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


def calculate_novelty(model, labeled_data, unlabeled_data, k=5):
    """Calcula scores de novelty usando KNN"""
    _, novelty_scores = novelty_detection(
        model, unlabeled_data, labeled_data, n_samples=len(unlabeled_data), k=k
    )
    return novelty_scores


def get_novel_indices(novelty_scores, threshold_percentile=75):
    """Retorna índices de imágenes novedosas/raras"""
    threshold = np.percentile(novelty_scores, threshold_percentile)
    is_novel = novelty_scores > threshold
    return np.where(is_novel)[0], is_novel


def visualize_novelty(images, labels, novelty_scores, is_novel, 
                      start_idx, output_dir='results'):
    """Genera visualizaciones de novelty analysis"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    num_images = len(images)
    num_graphs = (num_images + 19) // 20
    
    for graph_num in range(num_graphs):
        fig, axes = plt.subplots(4, 5, figsize=(18, 14))
        fig.suptitle(f'Novelty Detection Analysis (KNN) - Graph {graph_num+1}/{num_graphs}\n'
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
            score = novelty_scores[idx]
            
            title = f'Real: {CLASS_NAMES[true_label]}\n'
            title += f'Novelty: {score:.3f}\n'
            
            if is_novel[idx]:
                title += 'NOVEL'
                color, edge_color, edge_width = 'red', 'red', 3
            else:
                title += 'KNOWN'
                color, edge_color, edge_width = 'green', 'green', 1
            
            ax.set_title(title, fontsize=8, color=color, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(edge_width)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/novelty_analysis_{graph_num+1}.png', dpi=150, bbox_inches='tight')
        plt.close()


def analyze_novelty(model_path, labeled_size, start_idx, end_idx, k_neighbors=5,
                    output_dir='results', verbose=True):
    """
    Análisis completo de novelty para un rango de imágenes
    
    Args:
        model_path: Ruta al modelo .keras
        labeled_size: Número de imágenes etiquetadas (desde índice 0)
        start_idx: Índice inicial del rango a analizar
        end_idx: Índice final del rango a analizar
        k_neighbors: Número de vecinos más cercanos para KNN
        output_dir: Directorio para guardar resultados
        verbose: Mostrar mensajes de progreso
        
    Returns:
        novel_indices: Índices de imágenes novedosas
        novelty_scores: Scores de novelty para cada imagen
    """
    if verbose:
        print(f"Loading data and model...")
    
    x_train, y_train = load_cifar10_filtered()
    model = keras.models.load_model(model_path)
    
    labeled_data = x_train[:labeled_size]
    unlabeled_data = x_train[start_idx:end_idx]
    labels = y_train[start_idx:end_idx]
    
    if verbose:
        print(f"Labeled dataset: {labeled_size} images")
        print(f"Analyzing {len(unlabeled_data)} images ({start_idx}-{end_idx})...")
        print(f"K-neighbors: {k_neighbors}")
    
    novelty_scores = calculate_novelty(model, labeled_data, unlabeled_data, k_neighbors)
    novel_indices, is_novel = get_novel_indices(novelty_scores, threshold_percentile=75)
    
    if verbose:
        print(f"Found {len(novel_indices)} novel images ({len(novel_indices)/len(unlabeled_data)*100:.1f}%)")
    
    # visualize_novelty(unlabeled_data, labels, novelty_scores, is_novel, start_idx, output_dir)
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    np.save(f'{output_dir}/novelty_indices.npy', novel_indices)
    
    if verbose:
        print(f"Results saved to {output_dir}/")
    
    return novel_indices, novelty_scores


def main():
    """Ejecución standalone con parámetros por defecto"""
    LABELED_SIZE = 500
    START_INDEX = 501
    END_INDEX = 1000
    K_NEIGHBORS = 5
    
    print("="*60)
    print("NOVELTY DETECTION ANALYSIS")
    print("="*60)
    print(f"Labeled dataset: first {LABELED_SIZE} images")
    print(f"Analyzing: images {START_INDEX}-{END_INDEX}")
    print(f"K-neighbors: {K_NEIGHBORS}")
    print()
    
    novel_indices, scores = analyze_novelty(
        model_path=MODEL_PATH,
        labeled_size=LABELED_SIZE,
        start_idx=START_INDEX,
        end_idx=END_INDEX,
        k_neighbors=K_NEIGHBORS,
        output_dir='results',
        verbose=True
    )
    
    print()
    print("="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Novelty scores - Min: {scores.min():.3f}, Max: {scores.max():.3f}, Mean: {scores.mean():.3f}")
    print(f"Threshold (75th percentile): {np.percentile(scores, 75):.3f}")
    print()
    print("Analysis completed!")


if __name__ == "__main__":
    main()