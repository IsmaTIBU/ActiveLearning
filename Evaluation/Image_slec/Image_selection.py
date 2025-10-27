"""
Encuentra imágenes que cumplen las 3 características:
1. Alta incertidumbre (Uncertainty)
2. Lejos del centroide (Diversity)
3. Novedosa/rara (Novelty)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys

sys.path.insert(0, '..')

# ========================================
# CONFIGURACIÓN
# ========================================

SELECTED_CLASSES = [0, 1, 8]
CLASS_NAMES = ['Airplane', 'Automobile', 'Ship']
START_INDEX = 501
END_INDEX = 1000

MODEL_PATH = '/models/500_train/best_model.keras'


# ========================================
# CARGAR DATOS
# ========================================

def load_data():
    """Carga CIFAR-10 filtrado"""
    
    print("📥 Cargando CIFAR-10...")
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
    
    return x_train, y_train.flatten()


# ========================================
# MAIN
# ========================================

def main():
    
    print("="*60)
    print("🔍 ANÁLISIS DE INTERSECCIÓN")
    print("="*60)
    print()
    
    # Verificar que existen los archivos
    files = {
        'uncertainty': '../Uncertainty/results/uncertainty_indices.npy',
        'diversity': '../Diversity/results/diversity_indices.npy',
        'novelty': '../Novelty/results/novelty_indices.npy'
    }
    
    missing = [name for name, path in files.items() if not Path(path).exists()]
    
    if missing:
        print("ERROR: Faltan archivos. Ejecuta primero:")
        for name in missing:
            print(f"   - python analyze_{name}.py")
        return
    
    print("✓ Cargando resultados previos...")
    
    # Cargar índices
    uncertainty_idx = set(np.load(files['uncertainty']))
    diversity_idx = set(np.load(files['diversity']))
    novelty_idx = set(np.load(files['novelty']))
    
    print(f"  Uncertainty: {len(uncertainty_idx)} imágenes")
    print(f"  Diversity:   {len(diversity_idx)} imágenes")
    print(f"  Novelty:     {len(novelty_idx)} imágenes")
    print()
    
    # Intersección de las 3
    intersection = uncertainty_idx & diversity_idx & novelty_idx
    
    print(f"INTERSECCIÓN (cumplen las 3): {len(intersection)} imágenes")
    print()
    
    if len(intersection) == 0:
        print("No hay imágenes que cumplan las 3 características")
        return
    
    # Cargar datos
    x_train, y_train = load_data()
    
    # Crear carpeta
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar imágenes y crear txt
    print(f"Guardando {len(intersection)} imágenes en: {output_dir}")
    
    with open(output_dir / 'info.txt', 'w') as f:
        f.write("IMAGENES QUE CUMPLEN LAS 3 CARACTERISTICAS\n")
        f.write("="*60 + "\n\n")
        f.write("1. Alta incertidumbre (Uncertainty)\n")
        f.write("2. Lejos del centroide (Diversity)\n")
        f.write("3. Novedosa/rara (Novelty)\n\n")
        f.write("="*60 + "\n\n")
        
        for idx in sorted(intersection):
            # Índice real en el dataset
            real_idx = START_INDEX + idx
            
            # Imagen y label
            image = x_train[real_idx]
            label = y_train[real_idx]
            class_name = CLASS_NAMES[label]
            
            # Guardar imagen
            filename = f'image_{real_idx:04d}_class_{label}_{class_name}.png'
            plt.imsave(output_dir / filename, image)
            
            # Escribir en txt
            f.write(f"Image: {filename}\n")
            f.write(f"  - Indice en dataset: {real_idx}\n")
            f.write(f"  - Clase: {label} ({class_name})\n")
            f.write(f"  - Cumple: Uncertainty + Diversity + Novelty\n")
            f.write("\n")
            
            print(f"  ✓ {filename}")
    
    print()
    print(f"Completado!")
    print(f"Archivos guardados en: {output_dir}")
    print(f"Información detallada: {output_dir}/info.txt")
    
    # Mostrar resumen por clase
    print()
    print("="*60)
    print("RESUMEN POR CLASE")
    print("="*60)
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = sum(1 for idx in intersection if y_train[START_INDEX + idx] == class_id)
        print(f"  {class_name:12s}: {count} imágenes")


if __name__ == "__main__":

    main()