"""
Análisis Visual de Novelty Detection
Muestra cómo se detectan imágenes raras/nuevas usando KNN
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys

# Importar funciones desde AL_functions.py
sys.path.insert(0, '../..')
from AL_functions import novelty_detection


# ========================================
# CONFIGURACIÓN
# ========================================

NUM_CLASSES = 3
SELECTED_CLASSES = [0, 1, 8]  # airplane, automobile, ship
CLASS_NAMES = ['Airplane', 'Automobile', 'Ship']

LABELED_SIZE = 500  # Primeras 500 para entrenar
START_INDEX = 500   # Empezar análisis desde 501
END_INDEX = 600     # Hasta 600 (100 imágenes)

K_NEIGHBORS = 5     # Número de vecinos más cercanos para KNN

MODEL_PATH = '../../models/500_train/best_model.keras'


# ========================================
# 1. CARGAR DATOS Y MODELO
# ========================================

def load_data():
    """Carga CIFAR-10 filtrado"""
    
    print("Cargando CIFAR-10...")
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
    
    print(f"✓ Dataset: {len(x_train)} imágenes")
    return x_train, y_train


def load_model():
    """Carga el modelo entrenado"""
    
    print(f"Cargando modelo: {MODEL_PATH}")
    
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: No se encuentra el modelo en {MODEL_PATH}")
        exit(1)
    
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Modelo cargado")
    return model


# ========================================
# 2. APLICAR NOVELTY DETECTION
# ========================================

def analyze_novelty(model, labeled_data, unlabeled_data, k=5):
    """
    Aplica novelty_detection y analiza los resultados
    
    Returns:
        novelty_scores: score de novedad para cada imagen
    """
    
    print(f"Aplicando Novelty Detection (K={k})...")
    
    # Usar novelty_detection de AL_functions.py
    _, novelty_scores = novelty_detection(
        model, unlabeled_data, labeled_data, n_samples=50, k=k
    )
    
    print(f"✓ Novelty detection completado")
    print(f"  Score = distancia media a los {k} vecinos más cercanos")
    
    return novelty_scores


# ========================================
# 3. VISUALIZACIÓN
# ========================================

def plot_novelty_analysis(images, labels, novelty_scores, start_idx):
    """
    Crea 5 gráficos, cada uno con 20 imágenes
    """
    
    print(f"Creando gráficos...")
    
    Path('results').mkdir(exist_ok=True)
    
    # Calcular umbral (percentil 75 = top 25% más novedosas)
    threshold = np.percentile(novelty_scores, 75)
    is_novel = novelty_scores > threshold
    
    # 5 gráficos de 20 imágenes cada uno
    for graph_num in range(5):
        
        fig, axes = plt.subplots(4, 5, figsize=(18, 14))
        fig.suptitle(f'Novelty Detection Analysis (KNN) - Gráfico {graph_num+1}/5\n'
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
            score = novelty_scores[idx]
            novel = is_novel[idx]
            
            # Título con info
            title = f'Real: {CLASS_NAMES[true_label]}\n'
            title += f'Novelty Score: {score:.3f}\n'
            
            # Estado
            if novel:
                title += 'NOVELTY'
                color = 'red'
                edge_color = 'red'
                edge_width = 3
            else:
                title += 'KNOWN'
                color = 'green'
                edge_color = 'green'
                edge_width = 1
            
            ax.set_title(title, fontsize=8, color=color, fontweight='bold')
            
            # Borde
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(edge_width)
        
        plt.tight_layout()
        filename = f'results/novelty_analysis_{graph_num+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Guardado: {filename}")
        plt.close()


# ========================================
# 4. ESTADÍSTICAS
# ========================================

def print_statistics(novelty_scores, labels, k):
    """Imprime estadísticas de novelty"""
    
    print()
    print("="*60)
    print("📊 ESTADÍSTICAS DE NOVELTY")
    print("="*60)
    
    # Scores
    print(f"\nNovelty Scores (distancia media a {k} vecinos):")
    print(f"  Min:  {novelty_scores.min():.3f}")
    print(f"  Max:  {novelty_scores.max():.3f}")
    print(f"  Mean: {novelty_scores.mean():.3f}")
    print(f"  Std:  {novelty_scores.std():.3f}")
    
    # Umbral
    threshold = np.percentile(novelty_scores, 75)
    is_novel = novelty_scores > threshold
    
    # Imágenes novedosas por clase
    print(f"\nImágenes NOVEDOSAS/RARAS (top 25%) por clase:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = (labels == class_id) & is_novel
        count = mask.sum()
        total = (labels == class_id).sum()
        print(f"  {class_name:12s}: {count:2d}/{total:2d} ({count/total*100:.1f}%)")
    
    print(f"\nTotal imágenes NOVEDOSAS: {is_novel.sum()}/100 ({is_novel.sum()}%)")
    print(f"Umbral usado: {threshold:.3f} (percentil 75)")
    
    print()
    print("💡 Interpretación:")
    print("  - Score ALTO → imagen MUY diferente a las 500 etiquetadas")
    print("  - Score BAJO → imagen similar a las ya vistas")
    print("  - 🆕 NOVEL/RARE = potencialmente interesante para etiquetar")


# ========================================
# 5. MAIN
# ========================================

def main():
    
    print("="*60)
    print("🔍 ANÁLISIS DE NOVELTY DETECTION")
    print("="*60)
    print()
    
    # Cargar datos y modelo
    x_train, y_train = load_data()
    model = load_model()
    
    print()
    print(f"📋 Configuración:")
    print(f"  - Dataset etiquetado: primeras {LABELED_SIZE} imágenes")
    print(f"  - Analizar: imágenes {START_INDEX}-{END_INDEX} (100 imágenes)")
    print(f"  - K vecinos más cercanos: {K_NEIGHBORS}")
    print()
    
    # Aplicar novelty detection usando AL_functions.py
    novelty_scores = analyze_novelty(
        model,
        x_train[:LABELED_SIZE],           # Labeled (500)
        x_train[START_INDEX:END_INDEX],   # Unlabeled (100)
        k=K_NEIGHBORS
    )
    
    # Crear gráficos
    print()
    plot_novelty_analysis(
        x_train[START_INDEX:END_INDEX],
        y_train[START_INDEX:END_INDEX].flatten(),
        novelty_scores,
        START_INDEX
    )
    
    # Estadísticas
    print_statistics(
        novelty_scores,
        y_train[START_INDEX:END_INDEX].flatten(),
        K_NEIGHBORS
    )
    
    print()
    print("✅ Análisis completado!")
    print(f"📁 Gráficos guardados en: results/novelty_analysis_*.png")

    threshold = np.percentile(novelty_scores, 75)
    is_novel = novelty_scores > threshold
    np.save('results/novelty_indices.npy', np.where(is_novel)[0])
    print("💾 Índices guardados: results/novelty_indices.npy")


if __name__ == "__main__":
    main()