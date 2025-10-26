"""
Análisis Visual de Uncertainty Sampling
Muestra cómo el algoritmo selecciona las imágenes más inciertas
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import sys

# Importar funciones desde AL_functions.py
sys.path.append('/root')
from AL_functions import uncertainty_sampling


# ========================================
# CONFIGURACIÓN
# ========================================

NUM_CLASSES = 3
SELECTED_CLASSES = [0, 1, 8]  # airplane, automobile, ship
CLASS_NAMES = ['Airplane', 'Automobile', 'Ship']

LABELED_SIZE = 500  # Primeras 500 para entrenar
START_INDEX = 500   # Empezar análisis desde 501
END_INDEX = 600     # Hasta 600 (100 imágenes)
TOP_N = 50          # Seleccionar las 50 más inciertas

MODEL_PATH = 'models/best_model.keras'  # o 'models/final_model.keras'


# ========================================
# 1. CARGAR DATOS Y MODELO
# ========================================

def load_data():
    """Carga CIFAR-10 filtrado"""
    
    print("📥 Cargando CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
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
    
    print(f"🧠 Cargando modelo: {MODEL_PATH}")
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ ERROR: No se encuentra el modelo en {MODEL_PATH}")
        print("   Ejecuta primero: python train.py")
        exit(1)
    
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Modelo cargado")
    return model


# ========================================
# 2. CALCULAR SCORES USANDO AL_functions.py
# ========================================

def calculate_uncertainty_scores(model, images):
    """
    Calcula entropy y margin usando las funciones de AL_functions.py
    
    Returns:
        probabilities: (N, 3) - probabilidades por clase
        entropy_scores: (N,) - entropía de cada imagen
        margin_scores: (N,) - margin de cada imagen
    """
    
    print(f"🎲 Calculando uncertainty scores...")
    
    # Calcular probabilidades manualmente para mostrarlas
    probabilities = model.predict(images, verbose=0, batch_size=32)
    
    # Usar funciones de AL_functions.py
    _, entropy_scores = uncertainty_sampling(model, images, n_samples=len(images), method='entropy')
    _, margin_scores = uncertainty_sampling(model, images, n_samples=len(images), method='margin')
    
    print(f"✓ Scores calculados para {len(images)} imágenes")
    
    return probabilities, entropy_scores, margin_scores


# ========================================
# 3. DETERMINAR CUÁLES SON MÁS INCIERTAS
# ========================================

def get_top_uncertain_mask(scores, top_percent=50):
    """
    Marca las imágenes más inciertas (top 50% por defecto)
    
    Returns:
        is_uncertain: array booleano (True si está en el top)
    """
    
    threshold = np.percentile(scores, 100 - top_percent)
    is_uncertain = scores >= threshold
    
    return is_uncertain


# ========================================
# 4. VISUALIZACIÓN
# ========================================

def plot_uncertainty_analysis(images, labels, probabilities, 
                              entropy_scores, margin_scores, 
                              is_high_entropy, is_high_margin,
                              start_idx):
    """
    Crea 5 gráficos, cada uno con 20 imágenes
    """
    
    print(f"📊 Creando gráficos...")
    
    Path('results').mkdir(exist_ok=True)
    
    # 5 gráficos de 20 imágenes cada uno
    for graph_num in range(5):
        
        fig, axes = plt.subplots(4, 5, figsize=(18, 14))
        fig.suptitle(f'Uncertainty Sampling Analysis - Gráfico {graph_num+1}/5\n'
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
            pred_label = np.argmax(probabilities[idx])
            
            probs = probabilities[idx]
            entropy_val = entropy_scores[idx]
            margin_val = margin_scores[idx]
            
            # Determinar si es incierta
            high_e = is_high_entropy[idx]
            high_m = is_high_margin[idx]
            
            # Emojis
            emoji_e = '🎯' if high_e else '✓'
            emoji_m = '🎯' if high_m else '✓'
            
            # Título con info
            title = f'Real: {CLASS_NAMES[true_label]}\n'
            title += f'Pred: {CLASS_NAMES[pred_label]}\n'
            title += f'Probs: [{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]\n'
            title += f'Entropy: {entropy_val:.3f} {emoji_e}\n'
            title += f'Margin: {margin_val:.3f} {emoji_m}'
            
            # Color del borde según si está seleccionada
            if high_e or high_m:
                ax.set_title(title, fontsize=8, color='red', fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            else:
                ax.set_title(title, fontsize=8, color='green')
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(1)
        
        plt.tight_layout()
        filename = f'results/uncertainty_analysis_{graph_num+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Guardado: {filename}")
        plt.close()


# ========================================
# 5. MAIN
# ========================================

def main():
    
    print("="*60)
    print("🔬 ANÁLISIS DE UNCERTAINTY SAMPLING")
    print("="*60)
    print()
    
    # Cargar datos y modelo
    x_train, y_train = load_data()
    model = load_model()
    
    print()
    print(f"📋 Configuración:")
    print(f"  - Dataset etiquetado: primeras {LABELED_SIZE} imágenes")
    print(f"  - Analizar: imágenes {START_INDEX}-{END_INDEX} (100 imágenes)")
    print()
    
    # Extraer imágenes a analizar (501-600)
    images_to_analyze = x_train[START_INDEX:END_INDEX]
    labels_to_analyze = y_train[START_INDEX:END_INDEX]
    
    # Calcular scores usando AL_functions.py
    probabilities, entropy_scores, margin_scores = calculate_uncertainty_scores(
        model, images_to_analyze
    )
    
    # Determinar cuáles son más inciertas (top 50%)
    print()
    print(f"🎯 Identificando imágenes más inciertas (top 50%)...")
    
    is_high_entropy = get_top_uncertain_mask(entropy_scores, top_percent=50)
    is_high_margin = get_top_uncertain_mask(margin_scores, top_percent=50)
    
    high_entropy_count = is_high_entropy.sum()
    high_margin_count = is_high_margin.sum()
    overlap = (is_high_entropy & is_high_margin).sum()
    
    print(f"  ✓ Alta Entropy: {high_entropy_count} imágenes")
    print(f"  ✓ Alto Margin: {high_margin_count} imágenes")
    print(f"  ✓ Overlap: {overlap} imágenes en común")
    
    # Crear gráficos
    print()
    plot_uncertainty_analysis(
        images_to_analyze, 
        labels_to_analyze,
        probabilities,
        entropy_scores,
        margin_scores,
        is_high_entropy,
        is_high_margin,
        START_INDEX
    )
    
    # Estadísticas finales
    print()
    print("="*60)
    print("📊 ESTADÍSTICAS")
    print("="*60)
    print(f"Entropy scores:")
    print(f"  Min: {entropy_scores.min():.3f}")
    print(f"  Max: {entropy_scores.max():.3f}")
    print(f"  Mean: {entropy_scores.mean():.3f}")
    print()
    print(f"Margin scores:")
    print(f"  Min: {margin_scores.min():.3f}")
    print(f"  Max: {margin_scores.max():.3f}")
    print(f"  Mean: {margin_scores.mean():.3f}")
    print()
    print("✅ Análisis completado!")
    print(f"📁 Gráficos guardados en: results/uncertainty_analysis_*.png")
    print()
    print("🎯 = Alta incertidumbre (top 50%)")
    print("✓ = Baja incertidumbre")


if __name__ == "__main__":
    main()