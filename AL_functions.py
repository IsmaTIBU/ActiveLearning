import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import entropy
from tensorflow import keras


# ========================================
# 1. UNCERTAINTY SAMPLING
# ========================================

def uncertainty_sampling(model, unlabeled_data, n_samples=50, method='entropy'):
    """Selecciona muestras donde el modelo tiene más dudas"""
    
    probabilities = model.predict(unlabeled_data, verbose=0, batch_size=128)
    
    if method == 'entropy':
        scores = entropy(probabilities.T)
        
    elif method == 'margin':
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        scores = 1 - margin
    
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    selected = np.argsort(scores)[-n_samples:][::-1]
    return selected, scores


# ========================================
# 2. DIVERSITY SAMPLING
# ========================================

def diversity_sampling(model, unlabeled_data, n_samples=50):
    """Selecciona muestras diversas usando K-Means"""
    
    feature_extractor = get_feature_extractor(model)
    features = feature_extractor.predict(unlabeled_data, verbose=0, batch_size=128)
    
    kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
    kmeans.fit(features)
    
    selected = []
    for i in range(n_samples):
        cluster_samples = np.where(kmeans.labels_ == i)[0]
        
        if len(cluster_samples) > 0:
            cluster_center = kmeans.cluster_centers_[i]
            distances = euclidean_distances(
                features[cluster_samples], 
                cluster_center.reshape(1, -1)
            ).flatten()
            closest = cluster_samples[np.argmin(distances)]
            selected.append(closest)
    
    selected = np.array(selected)
    scores = np.zeros(len(unlabeled_data))
    scores[selected] = 1.0
    
    return selected, scores


# ========================================
# 3. NOVELTY DETECTION
# ========================================

def novelty_detection(model, unlabeled_data, labeled_data, n_samples=50, k=5):
    """Selecciona muestras raras usando KNN"""
    
    feature_extractor = get_feature_extractor(model)
    unlabeled_features = feature_extractor.predict(unlabeled_data, verbose=0, batch_size=128)
    labeled_features = feature_extractor.predict(labeled_data, verbose=0, batch_size=128)
    
    distances = euclidean_distances(unlabeled_features, labeled_features)
    k_nearest = np.sort(distances, axis=1)[:, :k]
    scores = k_nearest.mean(axis=1)
    
    selected = np.argsort(scores)[-n_samples:][::-1]
    return selected, scores


# ========================================
# 4. ESTRATEGIA COMBINADA
# ========================================

def combined_sampling(model, unlabeled_data, labeled_data=None, n_samples=50,
                     weights={'uncertainty': 0.5, 'diversity': 0.3, 'novelty': 0.2}):
    """Combina las 3 heurísticas"""
    
    _, unc_scores = uncertainty_sampling(model, unlabeled_data, n_samples)
    _, div_scores = diversity_sampling(model, unlabeled_data, n_samples)
    
    # Normalizar a [0, 1]
    unc_scores = (unc_scores - unc_scores.min()) / (unc_scores.max() - unc_scores.min() + 1e-10)
    div_scores = (div_scores - div_scores.min()) / (div_scores.max() - div_scores.min() + 1e-10)
    
    if labeled_data is not None and len(labeled_data) > 0:
        _, nov_scores = novelty_detection(model, unlabeled_data, labeled_data, n_samples)
        nov_scores = (nov_scores - nov_scores.min()) / (nov_scores.max() - nov_scores.min() + 1e-10)
    else:
        nov_scores = np.zeros_like(unc_scores)
        weights['novelty'] = 0.0
        total = weights['uncertainty'] + weights['diversity']
        weights['uncertainty'] /= total
        weights['diversity'] /= total
    
    combined = (
        weights['uncertainty'] * unc_scores +
        weights['diversity'] * div_scores +
        weights['novelty'] * nov_scores
    )
    
    selected = np.argsort(combined)[-n_samples:][::-1]
    return selected, combined


# ========================================
# 5. RANDOM BASELINE
# ========================================

def random_sampling(unlabeled_data, n_samples=50):
    """Selección aleatoria (baseline)"""
    return np.random.choice(len(unlabeled_data), size=n_samples, replace=False)


# ========================================
# 6. UTILIDADES
# ========================================

def get_feature_extractor(model):
    """Extrae features del modelo (penúltima capa)"""
    
    # Encontrar la capa de features (antes de Dropout y Dense final)
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if 'dropout' not in layer.name.lower() and 'dense' not in layer.name.lower():
            feature_layer = layer
            break
    else:
        feature_layer = model.layers[-2]
    
    # Para modelos Sequential, usar la entrada del primer layer
    try:
        return keras.Model(inputs=model.input, outputs=feature_layer.output)
    except AttributeError:
        # Si model.input falla, construir usando layers
        return keras.Model(inputs=model.layers[0].input, outputs=feature_layer.output)


# ========================================
# 7. FUNCIÓN PRINCIPAL
# ========================================

def select_samples(model, unlabeled_data, labeled_data=None, n_samples=50, 
                   strategy='uncertainty', **kwargs):
    """
    Selecciona muestras usando la estrategia elegida
    
    strategy: 'uncertainty', 'diversity', 'novelty', 'combined', 'random'
    """
    
    if strategy == 'uncertainty':
        selected, _ = uncertainty_sampling(model, unlabeled_data, n_samples, **kwargs)
        
    elif strategy == 'diversity':
        selected, _ = diversity_sampling(model, unlabeled_data, n_samples)
        
    elif strategy == 'novelty':
        if labeled_data is None:
            raise ValueError("novelty requiere labeled_data")
        selected, _ = novelty_detection(model, unlabeled_data, labeled_data, n_samples, **kwargs)
        
    elif strategy == 'combined':
        selected, _ = combined_sampling(model, unlabeled_data, labeled_data, n_samples, **kwargs)
        
    elif strategy == 'random':
        selected = random_sampling(unlabeled_data, n_samples)
        
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")
    
    return selected