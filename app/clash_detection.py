import numpy as np
import ifcopenshell
import ifcopenshell.geom
from scipy.spatial import KDTree
import time

class FastClashDetector:
    def __init__(self, tolerance=0.01, use_ai=True, debug=False):
        self.tolerance = max(tolerance, 0.001)  # Minimum 1mm
        self.use_ai = use_ai
        self.debug = debug
        
    def load_model(self, ifc_path):
        """Charge le modèle avec vérification des données"""
        model = ifcopenshell.open(ifc_path)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        
        elements = []
        for element in model.by_type("IfcElement"):
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                
                if len(verts) < 3:  # Au moins un triangle
                    continue
                    
                bbox_min = np.min(verts, axis=0)
                bbox_max = np.max(verts, axis=0)
                
                # Vérification des NaN
                if np.any(np.isnan(bbox_min)) or np.any(np.isnan(bbox_max)):
                    continue
                
                elements.append({
                    'element': element,
                    'guid': element.GlobalId,
                    'type': element.is_a(),
                    'name': getattr(element, 'Name', '') or element.GlobalId,
                    'bbox_min': bbox_min,
                    'bbox_max': bbox_max,
                    'center': (bbox_min + bbox_max) / 2,
                    'volume': np.prod(bbox_max - bbox_min)
                })
                
                if self.debug:
                    print(f"Element {element.GlobalId}: {element.is_a()}")
                    print(f"  BBox: {bbox_min} to {bbox_max}")
                    print(f"  Volume: {np.prod(bbox_max - bbox_min)}")
                    
            except Exception as e:
                print(f"Error processing element {element.GlobalId}: {str(e)}")
                continue
                
        print(f"Loaded {len(elements)} valid elements")
        return self._filter_elements(elements) if self.use_ai else elements
    
    def _filter_elements(self, elements):
        """Filtrage plus intelligent avec vérification des données"""
        if not elements:
            return []
            
        # Calcul des volumes et vérification
        volumes = np.array([e['volume'] for e in elements])
        if len(volumes) == 0:
            return []
            
        # Réduction du seuil de filtrage pour être moins agressif
        vol_threshold = np.percentile(volumes, 10)  # Plus permissif (10% au lieu de 20%)
        
        if self.debug:
            print(f"Volume threshold: {vol_threshold}")
            print(f"Volume range: {np.min(volumes)} to {np.max(volumes)}")
        
        filtered = []
        for elem in elements:
            # Vérification de la taille de la bbox - plus permissif
            bbox_size = elem['bbox_max'] - elem['bbox_min']
            if np.any(bbox_size < 0.0001):  # Réduit de 0.001 à 0.0001
                if self.debug:
                    print(f"Filtered out {elem['guid']} - too flat: {bbox_size}")
                continue
                
            # Seuil de volume plus permissif
            if elem['volume'] < vol_threshold * 0.1:  # Réduit de 0.5 à 0.1
                if self.debug:
                    print(f"Filtered out {elem['guid']} - too small: {elem['volume']}")
                continue
                
            filtered.append(elem)
        
        print(f"Filtered to {len(filtered)} elements")
        return filtered
    
    def detect_clashes(self, elements):
        """Détection avec plus de vérifications et logging"""
        if len(elements) < 2:
            print("Not enough elements for clash detection")
            return []

        # Vérification des coordonnées
        coords = np.array([e['center'] for e in elements])
        if np.any(np.isnan(coords)):
            print("NaN detected in element coordinates")
            return []
            
        if self.debug:
            print(f"Element centers:")
            for i, elem in enumerate(elements):
                print(f"  {elem['guid']}: {elem['center']}")
            
        try:
            kdtree = KDTree(coords)
        except Exception as e:
            print(f"KDTree creation failed: {str(e)}")
            return []
        
        clash_pairs = set()
        # Augmentation significative du rayon de recherche
        search_radius = max(self.tolerance * 20, 1.0)  # Au moins 1 mètre
        
        if self.debug:
            print(f"Search radius: {search_radius}")
        
        potential_pairs = 0
        actual_intersections = 0
        
        for i, elem in enumerate(elements):
            try:
                neighbors = kdtree.query_ball_point(elem['center'], search_radius)
                
                if self.debug and len(neighbors) > 1:
                    print(f"Element {elem['guid']} has {len(neighbors)-1} potential neighbors")
                
                for idx in neighbors:
                    if i == idx:
                        continue
                        
                    other = elements[idx]
                    pair = tuple(sorted((elem['guid'], other['guid'])))
                    
                    if pair not in clash_pairs:
                        potential_pairs += 1
                        if self._check_intersection(elem, other):
                            actual_intersections += 1
                            clash_pairs.add(pair)
                            if self.debug:
                                print(f"CLASH FOUND: {elem['type']} vs {other['type']}")
                                print(f"  Distance: {np.linalg.norm(elem['center'] - other['center'])}")
                        elif self.debug:
                            distance = np.linalg.norm(elem['center'] - other['center'])
                            print(f"No intersection: {elem['guid']} vs {other['guid']} (distance: {distance})")
                            
            except Exception as e:
                print(f"Error checking element {elem['guid']}: {str(e)}")
                continue

        if self.debug:
            print(f"Checked {potential_pairs} potential pairs, found {actual_intersections} intersections")

        report = self._generate_report(elements, clash_pairs)
        print(f"Found {len(report)} clashes")
        return report
    
    def _check_intersection(self, a, b):
        """Vérification plus robuste avec différents niveaux de tolérance"""
        # Test 1: Intersection stricte des bounding boxes
        strict_intersection = not (
            np.any(a['bbox_min'] > b['bbox_max']) or 
            np.any(b['bbox_min'] > a['bbox_max'])
        )
        
        if strict_intersection:
            if self.debug:
                print(f"  Strict intersection found")
            return True
        
        # Test 2: Intersection avec tolérance
        margin = self.tolerance
        tolerant_intersection = not (
            np.any(a['bbox_min'] > b['bbox_max'] + margin) or 
            np.any(b['bbox_min'] > a['bbox_max'] + margin)
        )
        
        if tolerant_intersection:
            if self.debug:
                print(f"  Tolerant intersection found (margin: {margin})")
            return True
        
        # Test 3: Proximité des centres (nouveau test)
        distance = np.linalg.norm(a['center'] - b['center'])
        avg_size = (np.mean(a['bbox_max'] - a['bbox_min']) + np.mean(b['bbox_max'] - b['bbox_min'])) / 2
        
        if distance < avg_size * 0.5:  # Si les centres sont très proches
            if self.debug:
                print(f"  Centers are close: distance={distance}, avg_size={avg_size}")
            return True
            
        return False
    
    def _generate_report(self, elements, clash_pairs):
        """Génération du rapport avec plus de détails"""
        guid_map = {e['guid']: e for e in elements}
        report = []
        
        for guid_a, guid_b in clash_pairs:
            try:
                a = guid_map[guid_a]
                b = guid_map[guid_b]
                
                intersection_min = np.maximum(a['bbox_min'], b['bbox_min'])
                intersection_max = np.minimum(a['bbox_max'], b['bbox_max'])
                intersection_size = np.maximum(0, intersection_max - intersection_min)
                
                overlap_vol = np.prod(intersection_size) if np.all(intersection_size > 0) else 0
                distance = np.linalg.norm(a['center'] - b['center'])
                
                report.append({
                    'element_a': {
                        'guid': a['guid'],
                        'name': a['name'],
                        'type': a['type'],
                        'bbox': {
                            'min': a['bbox_min'].tolist(),
                            'max': a['bbox_max'].tolist()
                        }
                    },
                    'element_b': {
                        'guid': b['guid'],
                        'name': b['name'],
                        'type': b['type'],
                        'bbox': {
                            'min': b['bbox_min'].tolist(),
                            'max': b['bbox_max'].tolist()
                        }
                    },
                    'distance': float(distance),
                    'overlap_volume': float(overlap_vol),
                    'intersection_size': intersection_size.tolist(),
                    'position': ((intersection_min + intersection_max) / 2).tolist() if np.all(intersection_size > 0) else a['center'].tolist()
                })
            except Exception as e:
                print(f"Error generating report for pair {guid_a}-{guid_b}: {str(e)}")
                continue
                
        return report