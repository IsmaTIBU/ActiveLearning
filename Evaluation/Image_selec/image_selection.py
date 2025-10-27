import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys


NUM_CLASSES = 3
SELECTED_CLASSES = [0, 1, 8]
CLASS_NAMES = ['Airplane', 'Automobile', 'Ship']


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


def load_indices_from_files(uncertainty_path, diversity_path, novelty_path):
    """Carga índices desde archivos .npy"""
    files = {
        'uncertainty': uncertainty_path,
        'diversity': diversity_path,
        'novelty': novelty_path
    }
    
    missing = [name for name, path in files.items() if not Path(path).exists()]
    
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")
    
    uncertainty_idx = set(np.load(uncertainty_path))
    diversity_idx = set(np.load(diversity_path))
    novelty_idx = set(np.load(novelty_path))
    
    return uncertainty_idx, diversity_idx, novelty_idx


def find_intersection(uncertainty_indices, diversity_indices, novelty_indices):
    """Encuentra la intersección de los tres conjuntos de índices"""
    unc_set = set(uncertainty_indices) if not isinstance(uncertainty_indices, set) else uncertainty_indices
    div_set = set(diversity_indices) if not isinstance(diversity_indices, set) else diversity_indices
    nov_set = set(novelty_indices) if not isinstance(novelty_indices, set) else novelty_indices
    
    intersection = unc_set & div_set & nov_set
    
    return sorted(list(intersection))


def save_intersection_images(intersection_indices, x_train, y_train, start_idx, 
                            output_dir='results', save_images=True, create_summary=True):
    """
    Guarda las imágenes de la intersección y crea un archivo de resumen
    
    Args:
        intersection_indices: Lista de índices relativos a start_idx
        x_train: Dataset completo de imágenes
        y_train: Labels del dataset
        start_idx: Índice inicial del rango analizado
        output_dir: Directorio de salida
        save_images: Si guardar las imágenes individuales
        create_summary: Si crear archivo de resumen
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    saved_files = []
    
    if save_images:
        for idx in intersection_indices:
            real_idx = start_idx + idx
            
            image = x_train[real_idx]
            label = y_train[real_idx]
            class_name = CLASS_NAMES[label]
            
            filename = f'image_{real_idx:04d}_class_{label}_{class_name}.png'
            plt.imsave(output_path / filename, image)
            saved_files.append(filename)
    
    if create_summary:
        with open(output_path / 'summary.txt', 'w') as f:
            f.write("IMAGES MEETING ALL 3 CRITERIA\n")
            f.write("="*60 + "\n\n")
            f.write("1. High uncertainty (Uncertainty Sampling)\n")
            f.write("2. Far from centroid (Diversity Sampling)\n")
            f.write("3. Novel/rare (Novelty Detection)\n\n")
            f.write("="*60 + "\n\n")
            
            for idx in intersection_indices:
                real_idx = start_idx + idx
                label = y_train[real_idx]
                class_name = CLASS_NAMES[label]
                
                f.write(f"Index: {real_idx}\n")
                f.write(f"  - Relative index: {idx}\n")
                f.write(f"  - Class: {label} ({class_name})\n")
                f.write(f"  - Criteria: Uncertainty + Diversity + Novelty\n")
                f.write("\n")
    
    return saved_files


def visualize_intersection(intersection_indices, x_train, y_train, start_idx,
                          output_dir='results', images_per_row=5):
    """
    Crea visualización de todas las imágenes de la intersección
    
    Args:
        intersection_indices: Lista de índices (relativos a start_idx)
        x_train, y_train: Datos
        start_idx: Índice inicial
        output_dir: Directorio de salida
        images_per_row: Imágenes por fila en la visualización
    """
    if len(intersection_indices) == 0:
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    num_images = len(intersection_indices)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row*3, num_rows*3))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Intersection Images - {num_images} images meeting all 3 criteria',
                fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(intersection_indices):
        row = i // images_per_row
        col = i % images_per_row
        ax = axes[row, col]
        
        real_idx = start_idx + idx
        image = x_train[real_idx]
        label = y_train[real_idx]
        
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Idx: {real_idx}\n{CLASS_NAMES[label]}', 
                    fontsize=10, fontweight='bold', color='red')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    
    for i in range(len(intersection_indices), num_rows * images_per_row):
        row = i // images_per_row
        col = i % images_per_row
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_file = output_path / 'intersection_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def get_class_distribution(intersection_indices, y_train, start_idx):
    """Calcula la distribución por clase de los índices de intersección"""
    distribution = {class_name: 0 for class_name in CLASS_NAMES}
    
    for idx in intersection_indices:
        real_idx = start_idx + idx
        label = y_train[real_idx]
        class_name = CLASS_NAMES[label]
        distribution[class_name] += 1
    
    return distribution


def analyze_intersection(uncertainty_indices, diversity_indices, novelty_indices,
                        start_idx, end_idx, output_dir='results',
                        save_images=True, create_visualization=True, verbose=True):
    """
    Análisis completo de intersección entre uncertainty, diversity y novelty
    
    Args:
        uncertainty_indices: Índices de uncertainty (relativos o absolutos)
        diversity_indices: Índices de diversity (relativos o absolutos)
        novelty_indices: Índices de novelty (relativos o absolutos)
        start_idx: Índice inicial del rango analizado
        end_idx: Índice final del rango analizado
        output_dir: Directorio para guardar resultados
        save_images: Si guardar imágenes individuales
        create_visualization: Si crear visualización combinada
        verbose: Mostrar mensajes de progreso
        
    Returns:
        intersection_indices: Lista de índices que cumplen los 3 criterios
        stats: Dict con estadísticas
    """
    if verbose:
        print("Loading CIFAR-10 dataset...")
    
    x_train, y_train = load_cifar10_filtered()
    
    if verbose:
        print(f"Finding intersection...")
        print(f"  Uncertainty: {len(uncertainty_indices)} images")
        print(f"  Diversity:   {len(diversity_indices)} images")
        print(f"  Novelty:     {len(novelty_indices)} images")
    
    intersection = find_intersection(uncertainty_indices, diversity_indices, novelty_indices)
    
    if verbose:
        print(f"  Intersection: {len(intersection)} images")
    
    if len(intersection) == 0:
        if verbose:
            print("No images meet all 3 criteria")
        return [], {'count': 0, 'distribution': {}}
    
    if save_images or verbose:
        saved_files = save_intersection_images(
            intersection, x_train, y_train, start_idx, 
            output_dir, save_images=save_images, create_summary=True
        )
        
        if verbose and save_images:
            print(f"Saved {len(saved_files)} images to {output_dir}/")
    
    if create_visualization:
        viz_file = visualize_intersection(
            intersection, x_train, y_train, start_idx, output_dir
        )
        if verbose and viz_file:
            print(f"Visualization saved: {viz_file}")
    
    distribution = get_class_distribution(intersection, y_train, start_idx)
    
    if verbose:
        print(f"\nClass distribution:")
        for class_name, count in distribution.items():
            print(f"  {class_name:12s}: {count} images")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    np.save(output_path / 'intersection_indices.npy', np.array(intersection))
    
    if verbose:
        print(f"\nIndices saved: {output_path}\intersection_indices.npy")
    
    stats = {
        'count': len(intersection),
        'distribution': distribution,
        'percentage': len(intersection) / (end_idx - start_idx) * 100 if end_idx > start_idx else 0
    }
    
    return intersection, stats


def analyze_intersection_from_files(uncertainty_file, diversity_file, novelty_file,
                                   start_idx, end_idx, output_dir='results', 
                                   save_images=True, create_visualization=True, verbose=True):
    """
    Análisis de intersección cargando índices desde archivos .npy
    
    Args:
        uncertainty_file: Path al archivo uncertainty_indices.npy
        diversity_file: Path al archivo diversity_indices.npy
        novelty_file: Path al archivo novelty_indices.npy
        start_idx: Índice inicial del rango analizado
        end_idx: Índice final del rango analizado
        output_dir: Directorio para resultados
        save_images: Si guardar imágenes
        create_visualization: Si crear visualización
        verbose: Mensajes de progreso
        
    Returns:
        intersection_indices: Lista de índices
        stats: Estadísticas
    """
    if verbose:
        print("Loading indices from files...")
    
    try:
        uncertainty_idx, diversity_idx, novelty_idx = load_indices_from_files(
            uncertainty_file, diversity_file, novelty_file
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run uncertainty, diversity, and novelty analyses first")
        return [], {'count': 0, 'distribution': {}}
    
    return analyze_intersection(
        uncertainty_idx, diversity_idx, novelty_idx,
        start_idx, end_idx, output_dir,
        save_images, create_visualization, verbose
    )


def main():
    """Ejecución standalone con parámetros por defecto"""
    START_INDEX = 501
    END_INDEX = 1000
    
    print("="*60)
    print("INTERSECTION ANALYSIS")
    print("="*60)
    print()
    
    uncertainty_file = '../Uncertainty/results/uncertainty_indices.npy'
    diversity_file = '../Diversity/results/diversity_indices.npy'
    novelty_file = '../Novelty/results/novelty_indices.npy'
    
    intersection, stats = analyze_intersection_from_files(
        uncertainty_file=uncertainty_file,
        diversity_file=diversity_file,
        novelty_file=novelty_file,
        start_idx=START_INDEX,
        end_idx=END_INDEX,
        output_dir='results',
        save_images=True,
        create_visualization=True,
        verbose=True
    )
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images analyzed: {END_INDEX - START_INDEX}")
    print(f"Images meeting all criteria: {stats['count']} ({stats['percentage']:.2f}%)")
    print()
    print("Analysis completed!")


if __name__ == "__main__":
    main()