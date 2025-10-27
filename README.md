# Active Learning Analysis - Refactored Library System

## Estructura del Sistema

El código ha sido refactorizado en **librerías modulares** que pueden:
1. **Ejecutarse independientemente** con parámetros por defecto
2. **Ser importadas** desde un programa centralizador con parámetros personalizados

```
.
├── uncertainty_lib.py            # Librería de uncertainty sampling
├── diversity_lib.py              # Librería de diversity sampling
├── novelty_lib.py                # Librería de novelty detection
├── image_selection_lib.py        # Librería de análisis de intersección
└── active_learning_analyzer.py   # Programa centralizador
```

---

## 1. Uso Independiente (Standalone)

Cada librería puede ejecutarse directamente con parámetros por defecto:

```bash
# Ejecutar uncertainty analysis
python uncertainty_lib.py

# Ejecutar diversity analysis
python diversity_lib.py

# Ejecutar novelty detection
python novelty_lib.py

# Ejecutar intersection analysis (requiere los 3 anteriores)
python image_selection_lib.py
```

**Parámetros por defecto en modo standalone:**
- `LABELED_SIZE = 500`
- `START_INDEX = 501`
- `END_INDEX = 1000`

---

## 2. Uso como Librería

### 2.1 Importación Básica

```python
from uncertainty_lib import analyze_uncertainty
from diversity_lib import analyze_diversity
from novelty_lib import analyze_novelty
from image_selection_lib import analyze_intersection

# Ejecutar análisis con parámetros personalizados
uncertain_indices, scores = analyze_uncertainty(
    model_path='path/to/model.keras',
    start_idx=100,
    end_idx=500,
    output_dir='my_results/uncertainty',
    threshold_percentile=60,
    verbose=True
)

diverse_indices, info = analyze_diversity(
    model_path='path/to/model.keras',
    start_idx=100,
    end_idx=500,
    n_clusters=30,
    output_dir='my_results/diversity',
    verbose=True
)

novel_indices, scores = analyze_novelty(
    model_path='path/to/model.keras',
    labeled_size=300,
    start_idx=100,
    end_idx=500,
    k_neighbors=10,
    output_dir='my_results/novelty',
    verbose=True
)

# Análisis de intersección
intersection, stats = analyze_intersection(
    uncertainty_indices=uncertain_indices,
    diversity_indices=diverse_indices,
    novelty_indices=novel_indices,
    start_idx=100,
    end_idx=500,
    output_dir='my_results/intersection',
    save_images=True,
    create_visualization=True,
    verbose=True
)
```

### 2.2 Parámetros Editables

#### `analyze_uncertainty()`
- `model_path`: Ruta al modelo .keras
- `start_idx`: Índice inicial del rango a analizar
- `end_idx`: Índice final del rango a analizar
- `output_dir`: Directorio para guardar resultados
- `threshold_percentile`: Percentil para considerar alta incertidumbre (default=50)
- `verbose`: Mostrar mensajes de progreso

**Returns:**
- `uncertain_indices`: np.array con índices de imágenes inciertas
- `scores`: Dict con `entropy`, `margin`, `probabilities`

#### `analyze_diversity()`
- `model_path`: Ruta al modelo .keras
- `start_idx`: Índice inicial
- `end_idx`: Índice final
- `n_clusters`: Número de clusters para K-Means (default=20)
- `output_dir`: Directorio de salida
- `verbose`: Verbosidad

**Returns:**
- `diverse_indices`: np.array con índices de imágenes diversas
- `info`: Dict con `cluster_labels`, `distances`, `selected_indices`

#### `analyze_novelty()`
- `model_path`: Ruta al modelo .keras
- `labeled_size`: Número de imágenes etiquetadas (desde índice 0)
- `start_idx`: Índice inicial
- `end_idx`: Índice final
- `k_neighbors`: Número de vecinos para KNN (default=5)
- `output_dir`: Directorio de salida
- `verbose`: Verbosidad

**Returns:**
- `novel_indices`: np.array con índices de imágenes novedosas
- `novelty_scores`: np.array con scores de novelty

#### `analyze_intersection()`
- `uncertainty_indices`: Índices de uncertainty (array o lista)
- `diversity_indices`: Índices de diversity (array o lista)
- `novelty_indices`: Índices de novelty (array o lista)
- `start_idx`: Índice inicial del rango analizado
- `end_idx`: Índice final del rango analizado
- `output_dir`: Directorio de salida
- `save_images`: Si guardar imágenes individuales (default=True)
- `create_visualization`: Si crear visualización combinada (default=True)
- `verbose`: Verbosidad

**Returns:**
- `intersection_indices`: Lista con índices que cumplen los 3 criterios
- `stats`: Dict con `count`, `distribution`, `percentage`

#### `analyze_intersection_from_files()`
Variante que carga índices desde archivos .npy:
- `uncertainty_file`: Path al archivo uncertainty_indices.npy
- `diversity_file`: Path al archivo diversity_indices.npy
- `novelty_file`: Path al archivo novelty_indices.npy
- (resto de parámetros igual que `analyze_intersection`)

---

## 3. Programa Centralizador

El programa `active_learning_analyzer.py` proporciona una interfaz unificada:

```python
from active_learning_analyzer import ActiveLearningAnalyzer

# Crear analizador con parámetros configurables
analyzer = ActiveLearningAnalyzer(
    model_path='models/500_train/best_model.keras',
    labeled_size=500,
    start_idx=501,
    end_idx=1000
)

# Opción 1: Ejecutar análisis individual
uncertain_indices, scores = analyzer.run_uncertainty(
    threshold_percentile=50,
    output_dir='results/uncertainty'
)

# Opción 2: Ejecutar todos los análisis
analyzer.run_all(
    uncertainty_params={'threshold_percentile': 50},
    diversity_params={'n_clusters': 20},
    novelty_params={'k_neighbors': 5}
)

# Encontrar intersección (imágenes que cumplen las 3 características)
intersection = analyzer.find_intersection(output_dir='results/intersection')

# Obtener resumen
print(analyzer.get_summary())
```

### 3.1 Métodos del ActiveLearningAnalyzer

```python
# Constructor
ActiveLearningAnalyzer(model_path, labeled_size, start_idx, end_idx)

# Ejecutar análisis individuales
analyzer.run_uncertainty(**params)
analyzer.run_diversity(**params)
analyzer.run_novelty(**params)

# Ejecutar todos
analyzer.run_all(uncertainty_params, diversity_params, novelty_params)

# Encontrar intersección
analyzer.find_intersection(output_dir)

# Obtener resumen
analyzer.get_summary()
```

---

## 4. Ejemplo Completo de Uso Centralizado

```python
from active_learning_analyzer import ActiveLearningAnalyzer

# Configurar parámetros centralizados
CONFIG = {
    'model_path': 'models/500_train/best_model.keras',
    'labeled_size': 500,
    'start_idx': 501,
    'end_idx': 1000
}

# Crear analizador
analyzer = ActiveLearningAnalyzer(**CONFIG)

# Ejecutar análisis completo
results = analyzer.run_all(
    uncertainty_params={
        'threshold_percentile': 50,
        'output_dir': 'results/uncertainty',
        'verbose': True
    },
    diversity_params={
        'n_clusters': 20,
        'output_dir': 'results/diversity',
        'verbose': True
    },
    novelty_params={
        'k_neighbors': 5,
        'output_dir': 'results/novelty',
        'verbose': True
    }
)

# Análisis de intersección
intersection = analyzer.find_intersection('results/intersection')

# Imprimir resumen
print(analyzer.get_summary())

# Acceder a resultados
print(f"Uncertain images: {len(results['uncertainty']['indices'])}")
print(f"Diverse images: {len(results['diversity']['indices'])}")
print(f"Novel images: {len(results['novelty']['indices'])}")
print(f"Intersection: {len(intersection)}")
```

---

## 5. Outputs Generados

Cada análisis genera:

### Archivos de Visualización
- `uncertainty_analysis_*.png` (1-5 gráficos de 20 imágenes)
- `diversity_analysis_*.png` (1-5 gráficos de 20 imágenes)
- `novelty_analysis_*.png` (1-5 gráficos de 20 imágenes)

### Archivos de Datos
- `uncertainty_indices.npy`: Índices de imágenes inciertas
- `diversity_indices.npy`: Índices de imágenes diversas
- `novelty_indices.npy`: Índices de imágenes novedosas
- `intersection_indices.npy`: Índices de intersección

### Archivos de Resumen
- `summary.txt`: Resumen de la intersección

---

## 6. Ventajas del Sistema Refactorizado

✅ **Modularidad**: Cada componente es independiente  
✅ **Reutilizabilidad**: Funciones pueden ser llamadas desde cualquier script  
✅ **Parametrización**: Todos los parámetros son configurables  
✅ **Centralización**: Un solo punto de control para múltiples análisis  
✅ **Flexibilidad**: Uso standalone o como librería  
✅ **Limpieza**: Sin prints innecesarios ni comentarios redundantes  

---

## 7. Migración desde Código Original

### Antes (Código Original)
```python
# Parámetros hardcodeados en el archivo
LABELED_SIZE = 500
START_INDEX = 501
END_INDEX = 1000

# Ejecutar script
python Evaluation/Uncertainty/uncertainty.py
```

### Ahora (Sistema Refactorizado)
```python
# Parámetros configurables desde código externo
from uncertainty_lib import analyze_uncertainty

uncertain_indices, scores = analyze_uncertainty(
    model_path='models/best_model.keras',
    start_idx=501,
    end_idx=1000,
    verbose=True
)
```

---

## 8. Notas Importantes

1. **Paths relativos**: Las librerías asumen que `AL_functions.py` está dos niveles arriba
2. **Modelos**: El modelo debe estar en formato `.keras`
3. **CIFAR-10**: Las librerías cargan y filtran CIFAR-10 automáticamente
4. **Verbosidad**: Control con `verbose=True/False` para silenciar outputs
5. **Resultados**: Los índices retornados son relativos al rango analizado (0-indexed)

---

## 9. Troubleshooting

**Error: "No module named AL_functions"**
```python
# Ajustar path si es necesario
sys.path.insert(0, 'path/to/AL_functions')
```

**Error: "Model not found"**
```python
# Verificar que el modelo existe
from pathlib import Path
if not Path(model_path).exists():
    print(f"Model not found: {model_path}")
```

**Memoria insuficiente**
```python
# Reducir batch_size en las funciones que predicen
# O reducir el rango de análisis (end_idx - start_idx)
```