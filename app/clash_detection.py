import numpy as np
import ifcopenshell
import ifcopenshell.geom
import time
import os.path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN  # Pour le regroupement intelligent des clashs

class FastIntraModelClashDetector:
    def __init__(self, tolerance=0.01, use_ai=True, debug_mode=False):
        self.tolerance = tolerance
        self.use_ai = use_ai
        self.debug_mode = debug_mode
        self.model = None
        self.elements = []
        
    def load_model(self, ifc_path):
        """Charge un modèle IFC et prépare les données pour la détection"""
        print(f"Chargement du modèle: {ifc_path}")
        self.model = ifcopenshell.open(ifc_path)
        self.elements = []
        
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        
        model_name = os.path.basename(ifc_path)
        valid_elements = 0
        skipped_elements = 0
        
        # Première passe: extraction rapide des bounding boxes
        for element in self.model.by_type("IfcElement"):
            try:
                if not element.Representation:
                    skipped_elements += 1
                    continue
                    
                shape = ifcopenshell.geom.create_shape(settings, element)
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                
                if len(verts) == 0 or np.any(np.isnan(verts)):
                    skipped_elements += 1
                    continue
                    
                min_coords = np.min(verts, axis=0)
                max_coords = np.max(verts, axis=0)
                center = np.mean(verts, axis=0)
                
                if np.any(np.isnan(min_coords)) or np.any(np.isnan(max_coords)):
                    skipped_elements += 1
                    continue
                    
                volume = np.prod(max_coords - min_coords)
                if volume < 1e-6:
                    skipped_elements += 1
                    continue
                
                self.elements.append({
                    'element': element,
                    'guid': element.GlobalId,
                    'name': getattr(element, 'Name', '') or element.GlobalId,
                    'type': element.is_a(),
                    'center': center,
                    'bbox_min': min_coords,
                    'bbox_max': max_coords,
                    'volume': volume
                })
                valid_elements += 1
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Erreur sur {element.GlobalId}: {str(e)}")
                skipped_elements += 1
                continue
        
        print(f"Modèle chargé: {valid_elements} éléments valides, {skipped_elements} ignorés")
        return valid_elements > 0
    
    def detect_clashes(self):
        """Détection de clash ultra-rapide avec optimisation IA"""
        if not self.elements:
            return []
            
        print(f"Début détection avec tolérance: {self.tolerance}m")
        t_start = time.time()
        
        # Filtrage IA des éléments pertinents
        relevant_elements = self._filter_elements_with_ai() if self.use_ai else self.elements
        
        # Construction du KDTree pour recherche spatiale accélérée
        positions = np.array([e['center'] for e in relevant_elements])
        kdtree = KDTree(positions)
        
        # Recherche des voisins proches
        clash_pairs = set()
        search_radius = self.tolerance * 3
        
        for i, elem_a in enumerate(relevant_elements):
            neighbors = kdtree.query_ball_point(elem_a['center'], search_radius)
            
            for idx in neighbors:
                if i == idx:
                    continue
                    
                elem_b = relevant_elements[idx]
                pair_id = tuple(sorted((elem_a['guid'], elem_b['guid'])))
                
                if pair_id in clash_pairs:
                    continue
                    
                if self._check_bbox_intersection(elem_a, elem_b, self.tolerance):
                    clash_pairs.add(pair_id)
        
        # Conversion en résultats détaillés
        clashes = []
        for guid_a, guid_b in clash_pairs:
            elem_a = next(e for e in relevant_elements if e['guid'] == guid_a)
            elem_b = next(e for e in relevant_elements if e['guid'] == guid_b)
            
            clash = self._create_clash_report(elem_a, elem_b)
            if clash:
                clashes.append(clash)
        
        # Regroupement intelligent des clashs avec DBSCAN
        if self.use_ai and clashes:
            clashes = self._cluster_clashes(clashes)
        
        t_end = time.time()
        print(f"Détection terminée: {len(clashes)} clashs trouvés en {t_end - t_start:.2f}s")
        return clashes
    
    def _filter_elements_with_ai(self):
        """Filtrage intelligent des éléments avec critères IA"""
        # 1. Éliminer les éléments trop petits (seuil dynamique)
        volumes = np.array([e['volume'] for e in self.elements])
        vol_threshold = np.percentile(volumes, 10)  # Seuil au 10ème percentile
        
        # 2. Prioriser certains types d'éléments
        priority_types = {
            'IfcDuct': 1.0, 'IfcPipe': 1.0, 'IfcCable': 1.0,
            'IfcBeam': 0.8, 'IfcColumn': 0.8, 'IfcWall': 0.6
        }
        
        filtered = []
        for elem in self.elements:
            # Critère de volume
            if elem['volume'] < vol_threshold * 0.1:  # 10% du seuil
                continue
                
            # Critère de type
            elem['priority'] = priority_types.get(elem['type'], 0.5)
            
            # Score combiné
            elem['score'] = np.log1p(elem['volume']) * elem['priority']
            filtered.append(elem)
        
        # Tri par score (meilleurs éléments en premier)
        filtered.sort(key=lambda x: -x['score'])
        
        # Limiter à 5000 éléments maximum pour performance
        return filtered[:5000]
    
    def _check_bbox_intersection(self, elem_a, elem_b, tolerance):
        """Vérification rapide d'intersection des bounding boxes"""
        min_a = elem_a['bbox_min'] - tolerance
        max_a = elem_a['bbox_max'] + tolerance
        
        min_b = elem_b['bbox_min'] - tolerance
        max_b = elem_b['bbox_max'] + tolerance
        
        return not (np.any(min_a > max_b) or np.any(min_b > max_a))
    
    def _create_clash_report(self, elem_a, elem_b):
        """Crée un rapport détaillé pour une paire en clash"""
        min_a = elem_a['bbox_min']
        max_a = elem_a['bbox_max']
        min_b = elem_b['bbox_min']
        max_b = elem_b['bbox_max']
        
        intersection_min = np.maximum(min_a, min_b)
        intersection_max = np.minimum(max_a, max_b)
        intersection_vol = np.prod(np.maximum(0, intersection_max - intersection_min))
        
        dist = np.linalg.norm(elem_a['center'] - elem_b['center'])
        rel_vol = intersection_vol / min(elem_a['volume'], elem_b['volume'])
        
        if rel_vol > 0.01 or dist < self.tolerance:
            return {
                'element_a': {
                    'guid': elem_a['guid'],
                    'name': elem_a['name'],
                    'type': elem_a['type'],
                    'bbox': [elem_a['bbox_min'].tolist(), elem_a['bbox_max'].tolist()]
                },
                'element_b': {
                    'guid': elem_b['guid'],
                    'name': elem_b['name'],
                    'type': elem_b['type'],
                    'bbox': [elem_b['bbox_min'].tolist(), elem_b['bbox_max'].tolist()]
                },
                'distance': float(dist),
                'overlap_volume': float(intersection_vol),
                'position': ((intersection_min + intersection_max) / 2).tolist()
            }
        return None
    
    def _cluster_clashes(self, clashes, eps=0.5, min_samples=2):
        """Regroupe les clashs similaires avec DBSCAN"""
        positions = np.array([c['position'] for c in clashes])
        
        # Normalisation des positions
        mean = np.mean(positions, axis=0)
        std = np.std(positions, axis=0)
        positions_norm = (positions - mean) / std
        
        # Clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(positions_norm)
        labels = db.labels_
        
        # Regroupement des clashs
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(clashes[i])
        
        # Sélection du clash le plus significatif par cluster
        representative_clashes = []
        for label, cluster in clusters.items():
            if label == -1:  # Bruit
                representative_clashes.extend(cluster)
            else:
                # Sélectionner le clash avec le plus grand volume d'intersection
                rep = max(cluster, key=lambda x: x['overlap_volume'])
                rep['cluster_size'] = len(cluster)
                representative_clashes.append(rep)
        
        return representative_clashes
    
    def get_element_geometry(self, element_guid):
        """Récupère la géométrie complète d'un élément"""
        element = next((e['element'] for e in self.elements if e['guid'] == element_guid), None)
        if not element:
            return None
            
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)
        faces = np.array(shape.geometry.faces).reshape(-1, 3)
        
        return {
            'vertices': verts.tolist(),
            'faces': faces.tolist(),
            'color': getattr(shape, 'styles', [])[0] if getattr(shape, 'styles', None) else [0.8, 0.8, 0.8]
        }