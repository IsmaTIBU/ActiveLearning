import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys

sys.path.insert(0, '../..')
from AL_functions import uncertainty_sampling


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


def calculate_uncertainty(model, images):
    """Calcula scores de uncertainty (entropy y margin)"""
    probabilities = model.predict(images, verbose=0, batch_size=128)
    _, entropy_scores = uncertainty_sampling(model, images, n_samples=len(images), method='entropy')
    _, margin_scores = uncertainty_sampling(model, images, n_samples=len(images), method='margin')
    
    return probabilities, entropy_scores, margin_scores


def get_uncertain_indices(entropy_scores, margin_scores, threshold_percentile=50):
    """Retorna índices de imágenes con alta incertidumbre"""
    entropy_threshold = np.percentile(entropy_scores, 100 - threshold_percentile)
    margin_threshold = np.percentile(margin_scores, 100 - threshold_percentile)
    
    is_high_entropy = entropy_scores >= entropy_threshold
    is_high_margin = margin_scores >= margin_threshold
    is_uncertain = is_high_entropy | is_high_margin
    
    return np.where(is_uncertain)[0], is_high_entropy, is_high_margin


def visualize_uncertainty(images, labels, probabilities, entropy_scores, margin_scores, 
                         is_high_entropy, is_high_margin, start_idx, output_dir='results'):
    """Genera visualizaciones de uncertainty analysis"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    num_images = len(images)
    num_graphs = (num_images + 19) // 20
    
    for graph_num in range(num_graphs):
        fig, axes = plt.subplots(4, 5, figsize=(18, 14))
        fig.suptitle(f'Uncertainty Sampling Analysis - Graph {graph_num+1}/{num_graphs}\n'
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
            pred_label = np.argmax(probabilities[idx])
            probs = probabilities[idx]
            
            title = f'Real: {CLASS_NAMES[true_label]}\n'
            title += f'Pred: {CLASS_NAMES[pred_label]}\n'
            title += f'Probs: [{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]\n'
            title += f'Entropy: {entropy_scores[idx]:.3f} {"HIGH" if is_high_entropy[idx] else "LOW"}\n'
            title += f'Margin: {margin_scores[idx]:.3f} {"HIGH" if is_high_margin[idx] else "LOW"}'
            
            color = 'red' if (is_high_entropy[idx] or is_high_margin[idx]) else 'green'
            edge_width = 3 if (is_high_entropy[idx] or is_high_margin[idx]) else 1
            
            ax.set_title(title, fontsize=8, color=color, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(edge_width)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/uncertainty_analysis_{graph_num+1}.png', dpi=150, bbox_inches='tight')
        plt.close()


def analyze_uncertainty(model_path, start_idx, end_idx, output_dir='results', 
                       threshold_percentile=50, verbose=True):
    """
    Análisis completo de uncertainty para un rango de imágenes
    
    Args:
        model_path: Ruta al modelo .keras
        start_idx: Índice inicial del rango a analizar
        end_idx: Índice final del rango a analizar
        output_dir: Directorio para guardar resultados
        threshold_percentile: Percentil para considerar alta incertidumbre
        verbose: Mostrar mensajes de progreso
        
    Returns:
        uncertain_indices: Índices de imágenes con alta incertidumbre
        scores: Dict con entropy_scores y margin_scores
    """
    if verbose:
        print(f"Loading data and model...")
    
    x_train, y_train = load_cifar10_filtered()
    model = keras.models.load_model(model_path)
    
    images = x_train[start_idx:end_idx]
    labels = y_train[start_idx:end_idx]
    
    if verbose:
        print(f"Analyzing {len(images)} images ({start_idx}-{end_idx})...")
    
    probabilities, entropy_scores, margin_scores = calculate_uncertainty(model, images)
    uncertain_indices, is_high_entropy, is_high_margin = get_uncertain_indices(
        entropy_scores, margin_scores, threshold_percentile
    )
    
    if verbose:
        print(f"Found {len(uncertain_indices)} uncertain images ({len(uncertain_indices)/len(images)*100:.1f}%)")
        print(f"  High entropy: {is_high_entropy.sum()}")
        print(f"  High margin: {is_high_margin.sum()}")
        print(f"  Overlap: {(is_high_entropy & is_high_margin).sum()}")
    
    # visualize_uncertainty(images, labels, probabilities, entropy_scores, margin_scores, is_high_entropy, is_high_margin, start_idx, output_dir)
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    np.save(f'{output_dir}/uncertainty_indices.npy', uncertain_indices)
    
    if verbose:
        print(f"Results saved to {output_dir}/")
    
    return uncertain_indices, {
        'entropy': entropy_scores,
        'margin': margin_scores,
        'probabilities': probabilities
    }


def main():
    """Ejecución standalone con parámetros por defecto"""
    LABELED_SIZE = 500
    START_INDEX = 501
    END_INDEX = 1000
    
    print("="*60)
    print("UNCERTAINTY SAMPLING ANALYSIS")
    print("="*60)
    print(f"Labeled dataset: first {LABELED_SIZE} images")
    print(f"Analyzing: images {START_INDEX}-{END_INDEX}")
    print()
    
    uncertain_indices, scores = analyze_uncertainty(
        model_path=MODEL_PATH,
        start_idx=START_INDEX,
        end_idx=END_INDEX,
        output_dir='results',
        threshold_percentile=50,
        verbose=True
    )
    
    print()
    print("="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Entropy - Min: {scores['entropy'].min():.3f}, Max: {scores['entropy'].max():.3f}, Mean: {scores['entropy'].mean():.3f}")
    print(f"Margin  - Min: {scores['margin'].min():.3f}, Max: {scores['margin'].max():.3f}, Mean: {scores['margin'].mean():.3f}")
    print()
    print("Analysis completed!")


if __name__ == "__main__":
    main()