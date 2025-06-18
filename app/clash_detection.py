import numpy as np
import ifcopenshell
import ifcopenshell.geom
from scipy.spatial import KDTree
from collections import defaultdict
import time

class FastClashDetector:
    def __init__(self, tolerance=0.01, use_ai=True):
        self.tolerance = tolerance
        self.use_ai = use_ai
        self.grid = defaultdict(list)
        
    def _get_cell_key(self, point, cell_size):
        return tuple((point // cell_size).astype(int))
    
    def _ai_prefilter(self, elements):
        """Filtrage IA des éléments les plus susceptibles de clash"""
        # Priorité aux éléments volumineux et de certains types
        type_weights = {
            'IfcDuct': 1.5, 'IfcPipe': 1.5, 'IfcCable': 1.3,
            'IfcBeam': 1.2, 'IfcColumn': 1.2, 'IfcWall': 1.0
        }
        
        for elem in elements:
            elem['priority'] = type_weights.get(elem['type'], 0.8)
            elem['score'] = np.log1p(elem['volume']) * elem['priority']
        
        return sorted(elements, key=lambda x: -x['score'])[:2000]  # Top 2000 éléments

    def load_model(self, ifc_path):
        """Chargement ultra-rapide avec prétraitement parallèle"""
        model = ifcopenshell.open(ifc_path)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        
        elements = []
        for element in model.by_type("IfcElement"):
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                
                if len(verts) == 0:
                    continue
                    
                bbox_min = np.min(verts, axis=0)
                bbox_max = np.max(verts, axis=0)
                
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
            except:
                continue
                
        return self._ai_prefilter(elements) if self.use_ai else elements

    def detect_clashes(self, elements):
        """Détection ultra-rapide avec grille spatiale et KDTree"""
        if len(elements) < 2:
            return []

        # Construction de la grille spatiale
        cell_size = max(self.tolerance * 5, 0.1)
        grid = defaultdict(list)
        
        for elem in elements:
            min_key = self._get_cell_key(elem['bbox_min'], cell_size)
            max_key = self._get_cell_key(elem['bbox_max'], cell_size)
            
            for x in range(min_key[0], max_key[0] + 1):
                for y in range(min_key[1], max_key[1] + 1):
                    for z in range(min_key[2], max_key[2] + 1):
                        grid[(x, y, z)].append(elem)

        # Détection avec KDTree
        positions = np.array([e['center'] for e in elements])
        kdtree = KDTree(positions)
        clash_pairs = set()

        for i, elem in enumerate(elements):
            neighbors = kdtree.query_ball_point(elem['center'], self.tolerance * 3)
            
            for idx in neighbors:
                if i == idx:
                    continue
                    
                other = elements[idx]
                pair = tuple(sorted((elem['guid'], other['guid'])))
                
                if pair not in clash_pairs and self._check_intersection(elem, other):
                    clash_pairs.add(pair)

        return self._generate_report(elements, clash_pairs)

    def _check_intersection(self, a, b):
        """Vérification rapide d'intersection"""
        return not (np.any(a['bbox_min'] > b['bbox_max']) or 
                    np.any(b['bbox_min'] > a['bbox_max']))

    def _generate_report(self, elements, clash_pairs):
        """Génération du rapport optimisée"""
        report = []
        guid_map = {e['guid']: e for e in elements}
        
        for guid_a, guid_b in clash_pairs:
            a = guid_map[guid_a]
            b = guid_map[guid_b]
            
            intersection_min = np.maximum(a['bbox_min'], b['bbox_min'])
            intersection_max = np.minimum(a['bbox_max'], b['bbox_max'])
            overlap_vol = np.prod(np.maximum(0, intersection_max - intersection_min))
            
            report.append({
                'element_a': {
                    'guid': a['guid'],
                    'name': a['name'],
                    'type': a['type'],
                    'bbox': [a['bbox_min'].tolist(), a['bbox_max'].tolist()]
                },
                'element_b': {
                    'guid': b['guid'],
                    'name': b['name'],
                    'type': b['type'],
                    'bbox': [b['bbox_min'].tolist(), b['bbox_max'].tolist()]
                },
                'distance': float(np.linalg.norm(a['center'] - b['center'])),
                'overlap_volume': float(overlap_vol),
                'position': ((intersection_min + intersection_max) / 2).tolist()
            })
            
        return report