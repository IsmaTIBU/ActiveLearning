"""
Programa centralizador para análisis de Active Learning
Permite ejecutar uncertainty, diversity y novelty de forma unificada
"""

import numpy as np
from pathlib import Path
import sys

from Evaluation.Uncertainty.uncertainty import analyze_uncertainty
from Evaluation.Diversity.diversity import analyze_diversity
from Evaluation.Novelty.novelty import analyze_novelty
from Evaluation.Image_selec.image_selection import analyze_intersection, CLASS_NAMES
from train import train_model


class ActiveLearningAnalyzer:
    """Centralizador de análisis de Active Learning"""
    
    def __init__(self, model_path, labeled_size=500, start_idx=501, end_idx=1000,
                 target_per_class=None):
        """
        Args:
            model_path: Ruta al modelo .keras
            labeled_size: Número de imágenes etiquetadas
            start_idx: Índice inicial del rango a analizar
            end_idx: Índice final del rango a analizar
            target_per_class: Dict o int con número objetivo de imágenes por clase
                             Ej: {'Airplane': 10, 'Automobile': 10, 'Ship': 10}
                             o simplemente: 10 (aplica a todas las clases)
        """
        self.model_path = model_path
        self.labeled_size = labeled_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        if target_per_class is not None:
            if isinstance(target_per_class, int):
                self.target_per_class = {cls: target_per_class for cls in CLASS_NAMES}
            else:
                self.target_per_class = target_per_class
        else:
            self.target_per_class = None
        
        self.results = {}
    
    def run_uncertainty(self, threshold_percentile=50, output_dir='results/uncertainty', verbose=True):
        """Ejecuta análisis de uncertainty"""
        if verbose:
            print("\n" + "="*60)
            print("UNCERTAINTY ANALYSIS")
            print("="*60)
        
        uncertain_indices, scores = analyze_uncertainty(
            model_path=self.model_path,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            output_dir=output_dir,
            threshold_percentile=threshold_percentile,
            verbose=verbose
        )
        
        self.results['uncertainty'] = {
            'indices': uncertain_indices,
            'scores': scores
        }
        
        return uncertain_indices, scores
    
    def run_diversity(self, n_clusters=20, output_dir='results/diversity', verbose=True):
        """Ejecuta análisis de diversity"""
        if verbose:
            print("\n" + "="*60)
            print("DIVERSITY ANALYSIS")
            print("="*60)
        
        diverse_indices, info = analyze_diversity(
            model_path=self.model_path,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            n_clusters=n_clusters,
            output_dir=output_dir,
            verbose=verbose
        )
        
        self.results['diversity'] = {
            'indices': diverse_indices,
            'info': info
        }
        
        return diverse_indices, info
    
    def run_novelty(self, k_neighbors=5, output_dir='results/novelty', verbose=True):
        """Ejecuta análisis de novelty"""
        if verbose:
            print("\n" + "="*60)
            print("NOVELTY ANALYSIS")
            print("="*60)
        
        novel_indices, scores = analyze_novelty(
            model_path=self.model_path,
            labeled_size=self.labeled_size,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            k_neighbors=k_neighbors,
            output_dir=output_dir,
            verbose=verbose
        )
        
        self.results['novelty'] = {
            'indices': novel_indices,
            'scores': scores
        }
        
        return novel_indices, scores
    
    def run_all(self, uncertainty_params=None, diversity_params=None, novelty_params=None):
        """Ejecuta los tres análisis"""
        print("\n" + "="*60)
        print("COMPLETE ACTIVE LEARNING ANALYSIS")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Labeled size: {self.labeled_size}")
        print(f"Analysis range: {self.start_idx}-{self.end_idx}")
        print(f"Total images: {self.end_idx - self.start_idx}")
        
        uncertainty_params = uncertainty_params or {}
        diversity_params = diversity_params or {}
        novelty_params = novelty_params or {}
        
        self.run_uncertainty(**uncertainty_params)
        self.run_diversity(**diversity_params)
        self.run_novelty(**novelty_params)
        
        return self.results
    
    def find_intersection(self, output_dir='results/intersection', 
                         save_images=True, create_visualization=True, verbose=True):
        """
        Encuentra imágenes que cumplen las 3 características usando image_selection_lib
        
        Args:
            output_dir: Directorio para guardar resultados
            save_images: Si guardar imágenes individuales
            create_visualization: Si crear visualización combinada
            verbose: Mostrar mensajes
            
        Returns:
            intersection_indices: Lista de índices
            stats: Estadísticas de la intersección
        """
        if not all(k in self.results for k in ['uncertainty', 'diversity', 'novelty']):
            raise ValueError("Must run all analyses first")
        
        if verbose:
            print("\n" + "="*60)
            print("INTERSECTION ANALYSIS")
            print("="*60)
        
        intersection, stats = analyze_intersection(
            uncertainty_indices=self.results['uncertainty']['indices'],
            diversity_indices=self.results['diversity']['indices'],
            novelty_indices=self.results['novelty']['indices'],
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            output_dir=output_dir,
            save_images=save_images,
            create_visualization=create_visualization,
            verbose=verbose
        )
        
        self.results['intersection'] = {
            'indices': intersection,
            'stats': stats
        }
        
        if self.target_per_class is not None:
            self._validate_class_balance(stats['distribution'])
        
        return intersection, stats
    
    def _validate_class_balance(self, distribution):
        """
        Valida si se alcanzó el objetivo de imágenes por clase
        Si se cumple, entrena un nuevo modelo con los datos adicionales
        
        Args:
            distribution: Dict con distribución actual por clase
        """
        print("\n" + "="*60)
        print("CLASS BALANCE VALIDATION")
        print("="*60)
        
        total_images = self.end_idx - self.start_idx
        missing_classes = []
        all_ok = True
        
        for class_name, target in self.target_per_class.items():
            actual = distribution.get(class_name, 0)
            status = "✓" if actual >= target else "✗"
            
            print(f"{status} {class_name:12s}: {actual}/{target} images")
            
            if actual < target:
                missing_classes.append((class_name, target - actual))
                all_ok = False
        
        print()
        
        if all_ok:
            print("✓ All class targets met!")
            print()
            print("="*60)
            print("RETRAINING MODEL WITH NEW DATA")
            print("="*60)
            
            intersection_indices = self.results['intersection']['indices']
            absolute_indices = np.array([self.start_idx + idx for idx in intersection_indices])
            
            model, history = train_model(
                additional_indices=absolute_indices,
                output_dir='models_retrained',
                verbose=True
            )
            
            print("\n✓ Model retrained successfully!")
            print("  Saved to: models_retrained/best_model.keras")
            
        else:
            print("✗ Insufficient images for some classes")
            print()
            print("RECOMMENDATIONS:")
            
            total_missing = sum(missing for _, missing in missing_classes)
            
            print(f"  Current range: {self.start_idx}-{self.end_idx} ({total_images} images)")
            print(f"  Missing images: {total_missing} total")
            print()
            print("  Options:")
            print(f"    1. Increase end_idx to analyze more images")
            print(f"       Suggested: end_idx >= {self.end_idx + total_missing * 3}")
            print(f"    2. Reduce target_per_class requirements")
            print(f"    3. Adjust Active Learning parameters (lower thresholds)")
            print()
            
            for class_name, missing in missing_classes:
                print(f"    - {class_name}: needs {missing} more images")
        
        print("="*60)
    
    def get_summary(self):
        """Retorna resumen de todos los análisis"""
        if not self.results:
            return "No analyses run yet"
        
        summary = []
        summary.append("="*60)
        summary.append("ACTIVE LEARNING ANALYSIS SUMMARY")
        summary.append("="*60)
        summary.append(f"Model: {self.model_path}")
        summary.append(f"Labeled: {self.labeled_size}, Range: {self.start_idx}-{self.end_idx}")
        summary.append("")
        
        if 'uncertainty' in self.results:
            unc = self.results['uncertainty']
            summary.append(f"UNCERTAINTY: {len(unc['indices'])} images")
            summary.append(f"  Entropy: {unc['scores']['entropy'].mean():.3f} ± {unc['scores']['entropy'].std():.3f}")
            summary.append(f"  Margin:  {unc['scores']['margin'].mean():.3f} ± {unc['scores']['margin'].std():.3f}")
        
        if 'diversity' in self.results:
            div = self.results['diversity']
            summary.append(f"DIVERSITY: {len(div['indices'])} images")
            summary.append(f"  Distance: {div['info']['distances'].mean():.3f} ± {div['info']['distances'].std():.3f}")
            summary.append(f"  Representatives: {len(div['info']['selected_indices'])}")
        
        if 'novelty' in self.results:
            nov = self.results['novelty']
            summary.append(f"NOVELTY: {len(nov['indices'])} images")
            summary.append(f"  Score: {nov['scores'].mean():.3f} ± {nov['scores'].std():.3f}")
        
        if 'intersection' in self.results:
            inter = self.results['intersection']
            summary.append(f"INTERSECTION: {inter['stats']['count']} images ({inter['stats']['percentage']:.2f}%)")
            summary.append(f"  Distribution: {inter['stats']['distribution']}")
        
        summary.append("="*60)
        
        return "\n".join(summary)


def main():
    """Ejemplo de uso del centralizador con objetivo de 50 imágenes por clase"""
    
    MODEL_PATH = 'models/500_train/best_model.keras'
    LABELED_SIZE = 500
    START_INDEX = 501
    END_INDEX = 4000
    
    analyzer = ActiveLearningAnalyzer(
        model_path=MODEL_PATH,
        labeled_size=LABELED_SIZE,
        start_idx=START_INDEX,
        end_idx=END_INDEX,
        target_per_class=50
    )
    
    analyzer.run_all(
        uncertainty_params={'threshold_percentile': 50},
        diversity_params={'n_clusters': 20},
        novelty_params={'k_neighbors': 5}
    )
    
    print(analyzer.get_summary())
    
    intersection, stats = analyzer.find_intersection(
        save_images=True,
        create_visualization=True
    )
    
    print(f"\nFound {len(intersection)} images meeting all criteria")


if __name__ == "__main__":
    main()