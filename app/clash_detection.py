import numpy as np
import ifcopenshell
import ifcopenshell.geom
import time
import os.path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        
    def _get_cell_key(self, point):
        return (
            int(point[0] // self.cell_size),
            int(point[1] // self.cell_size),
            int(point[2] // self.cell_size)
        )
    
    def insert(self, element):
        min_coord = element['bbox_min']
        max_coord = element['bbox_max']
        
        min_cell = self._get_cell_key(min_coord)
        max_cell = self._get_cell_key(max_coord)
        
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    self.grid[(x,y,z)].append(element)

class FastClashDetector:
    def __init__(self, tolerance=0.01, debug_mode=True):
        self.tolerance = tolerance
        self.debug_mode = debug_mode
        self.models = []
        
    def add_model(self, ifc_path):
        """Charge un modèle IFC et prépare les données pour la détection"""
        print(f"Chargement du modèle: {ifc_path}")
        model = ifcopenshell.open(ifc_path)
        elements = []
        
        # Configuration pour la géométrie
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        
        # Extraction de la géométrie des éléments
        model_name = os.path.basename(ifc_path)
        valid_elements = 0
        skipped_elements = 0
        
        for element in model.by_type("IfcElement"):
            try:
                if not element.Representation:
                    if self.debug_mode:
                        print(f"Element {element.GlobalId} n'a pas de représentation géométrique")
                    skipped_elements += 1
                    continue
                shape = ifcopenshell.geom.create_shape(settings, element)
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                if len(verts) == 0 or np.any(np.isnan(verts)):
                    raise ValueError("Vertices invalides")
                # Si des vertices sont trouvés, enregistrer l'élément
                if len(verts) > 0:
                    min_coords = np.min(verts, axis=0)
                    max_coords = np.max(verts, axis=0)
                    center = np.mean(verts, axis=0)
                    
                    # Vérifier la validité de la boîte englobante
                    if np.any(np.isnan(min_coords)) or np.any(np.isnan(max_coords)):
                        if self.debug_mode:
                            print(f"Avertissement: Boîte englobante invalide pour {element.GlobalId}")
                        skipped_elements += 1
                        continue
                        
                    # Calculer le volume de la boîte englobante
                    volume = np.prod(max_coords - min_coords)
                    if volume < 1e-6:  # Volume très petit, probablement un point ou une ligne
                        if self.debug_mode:
                            print(f"Avertissement: Élément trop petit ignoré {element.GlobalId}")
                        skipped_elements += 1
                        continue
                    
                    elements.append({
                        'guid': element.GlobalId,
                        'name': getattr(element, 'Name', '') or element.GlobalId,
                        'type': element.is_a(),
                        'center': center,
                        'bbox_min': min_coords,
                        'bbox_max': max_coords,
                        'volume': volume,
                        'path': ifc_path,
                        'model_name': model_name
                    })
                    valid_elements += 1
            except Exception as e:
                print(f"Erreur lors du traitement de l'élément {element.GlobalId}: {str(e)}")
                skipped_elements += 1
                continue
        
        print(f"Modèle {model_name} chargé: {valid_elements} éléments valides, {skipped_elements} éléments ignorés")
        
        if valid_elements > 0:
            # Calculer les limites du modèle pour analyse de coordonnées
            all_centers = np.array([e['center'] for e in elements])
            min_model = np.min(all_centers, axis=0)
            max_model = np.max(all_centers, axis=0)
            
            print(f"Limites du modèle {model_name}:")
            print(f"  X: {min_model[0]:.2f} à {max_model[0]:.2f}")
            print(f"  Y: {min_model[1]:.2f} à {max_model[1]:.2f}")
            print(f"  Z: {min_model[2]:.2f} à {max_model[2]:.2f}")
            
            model_info = {
                'path': ifc_path,
                'elements': elements,
                'model_name': model_name,
                'bounds_min': min_model,
                'bounds_max': max_model
            }
            
            self.models.append(model_info)
            return model_info
        else:
            print(f"Avertissement: Aucun élément valide dans {model_name}")
            return None

    def get_element_geometry(self, element):
        """Extrait la géométrie détaillée d'un élément"""
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

    def detect_clashes(self, auto_adjust_tolerance=True):
        """Détection de clash entre tous les modèles avec ajustement automatique de tolérance"""
        if len(self.models) < 2:
            print("Au moins deux modèles sont nécessaires pour la détection")
            return []
        
        print("Analyse des modèles avant détection...")
        
        # Vérifier si les modèles sont dans le même système de coordonnées
        self._check_coordinate_systems()
        
        # Ajuster la tolérance en fonction de la taille des modèles si requis
        if auto_adjust_tolerance:
            self._adjust_tolerance()
        
        print(f"Début de la détection des clashs (tolérance: {self.tolerance}m)...")
        t_start = time.time()
        clashes = []
        
        # Comparaison entre tous les modèles chargés
        for i in range(len(self.models)):
            for j in range(i+1, len(self.models)):
                model_a = self.models[i]
                model_b = self.models[j]
                
                print(f"Comparaison entre {model_a['model_name']} et {model_b['model_name']}")
                
                # Vérifier rapidement si les modèles se chevauchent globalement
                if not self._check_models_overlap(model_a, model_b, self.tolerance * 5):
                    print(f"  → Les modèles ne se chevauchent pas, détection ignorée")
                    continue
                
                # Détection entre les deux modèles
                found_clashes = self._detect_between_models(model_a, model_b)
                clashes.extend(found_clashes)
                
                print(f"  → {len(found_clashes)} clashs trouvés")
        
        t_end = time.time()
        print(f"Détection terminée. {len(clashes)} clashs trouvés en {t_end - t_start:.2f} secondes")
        return clashes

    def detect_intra_model_clashes(self, model_info):
    elements = model_info['elements']
    
    # Optimisation: Filtrer les petits éléments non pertinents
    relevant_elements = [
        elem for elem in elements 
        if elem['volume'] > 0.001  # Ignorer les éléments très petits
    ]
    
    # Utiliser un KDTree pour une recherche spatiale rapide
    from scipy.spatial import KDTree
    positions = np.array([e['center'] for e in relevant_elements])
    kdtree = KDTree(positions)
    
    clashes = []
    processed_pairs = set()
    
    # Réduire la tolérance initiale pour moins de paires
    search_radius = self.tolerance * 5
    
    for i, elem_a in enumerate(relevant_elements):
        # Trouver les voisins proches
        neighbors = kdtree.query_ball_point(elem_a['center'], search_radius)
        
        for idx in neighbors:
            if i == idx:
                continue
                
            elem_b = relevant_elements[idx]
            pair_id = tuple(sorted((elem_a['guid'], elem_b['guid'])))
            
            if pair_id in processed_pairs:
                continue
                
            processed_pairs.add(pair_id)
            
            # Vérification rapide des boîtes englobantes
            if self._quick_bbox_proximity(elem_a, elem_b, self.tolerance * 3):
                clash = self._check_detailed_clash(elem_a, elem_b, self.tolerance)
                if clash:
                    clashes.append(clash)
    
    return clashes
    def _get_neighbors(self, grid, element):
        """Récupère les éléments voisins d'un élément donné via la grille spatiale"""
        neighbors = set()
        cell_keys = self._get_cell_keys_for_element(element, grid.cell_size)
        
        for key in cell_keys:
            if key in grid.grid:
                for neighbor in grid.grid[key]:
                    # Ne pas ajouter l'élément lui-même
                    if neighbor['guid'] != element['guid']:
                        neighbors.add(neighbor['guid'])  # Use GUID as key to avoid duplicates
        
        # Convert back to element objects
        return [elem for elem in grid.grid[cell_keys[0]] if elem['guid'] in neighbors] if cell_keys else []

    def _get_cell_keys_for_element(self, element, cell_size):
        """Génère toutes les clés de cellule couvertes par un élément"""
        min_coord = element['bbox_min']
        max_coord = element['bbox_max']
        
        min_cell = (
            int(min_coord[0] // cell_size),
            int(min_coord[1] // cell_size),
            int(min_coord[2] // cell_size)
        )
        max_cell = (
            int(max_coord[0] // cell_size),
            int(max_coord[1] // cell_size),
            int(max_coord[2] // cell_size)
        )
        
        keys = []
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    keys.append((x, y, z))
        return keys
    
    def _check_coordinate_systems(self):
        """Vérifie si les modèles semblent être dans le même système de coordonnées"""
        if len(self.models) < 2:
            return
            
        # Comparer les limites des modèles
        first_min = self.models[0]['bounds_min']
        first_max = self.models[0]['bounds_max']
        
        for i, model in enumerate(self.models[1:], 1):
            model_min = model['bounds_min']
            model_max = model['bounds_max']
            
            # Calculer la distance entre les centres des modèles
            center1 = (first_max + first_min) / 2
            center2 = (model_max + model_min) / 2
            distance = np.linalg.norm(center2 - center1)
            
            # Calculer la diagonale du premier modèle
            diagonal = np.linalg.norm(first_max - first_min)
            
            # Si la distance est très grande par rapport à la taille du modèle
            if distance > diagonal * 100:
                print(f"AVERTISSEMENT: Les modèles {self.models[0]['model_name']} et {model['model_name']} semblent utiliser des systèmes de coordonnées différents!")
                print(f"  Distance entre les centres: {distance:.2f}m (diagonale modèle 1: {diagonal:.2f}m)")
                print(f"  Cela peut empêcher la détection correcte des clashs.")
    
    def _adjust_tolerance(self):
        """Ajuste la tolérance en fonction de la taille des modèles"""
        # Calculer la diagonale moyenne des modèles
        diagonals = []
        for model in self.models:
            diag = np.linalg.norm(model['bounds_max'] - model['bounds_min'])
            diagonals.append(diag)
        
        avg_diagonal = np.mean(diagonals)
        
        # Ajuster la tolérance à environ 0.1% de la diagonale moyenne
        suggested_tolerance = avg_diagonal * 0.001
        
        # Limiter à des valeurs raisonnables
        if suggested_tolerance < 0.005:
            suggested_tolerance = 0.005  # Minimum 5mm
        elif suggested_tolerance > 0.1:
            suggested_tolerance = 0.1    # Maximum 10cm
            
        if abs(suggested_tolerance - self.tolerance) > 0.001:
            old_tolerance = self.tolerance
            self.tolerance = suggested_tolerance
            print(f"Tolérance ajustée de {old_tolerance:.3f}m à {self.tolerance:.3f}m (basée sur la taille des modèles)")
    
    def _check_models_overlap(self, model_a, model_b, extra_margin=0.0):
        """Vérifie rapidement si deux modèles se chevauchent globalement"""
        min_a = model_a['bounds_min'] - extra_margin
        max_a = model_a['bounds_max'] + extra_margin
        
        min_b = model_b['bounds_min'] - extra_margin
        max_b = model_b['bounds_max'] + extra_margin
        
        # Vérifier si les boîtes englobantes s'intersectent
        if np.any(min_a > max_b) or np.any(min_b > max_a):
            return False  # Pas d'intersection
        
        return True  # Les modèles se chevauchent potentiellement
    
    def _detect_between_models(self, model_a, model_b):
        """Détecte les clashs entre deux modèles spécifiques avec une meilleure stratégie"""
        elements_a = model_a['elements']
        elements_b = model_b['elements']
        
        model_clashes = []
        
        # Optimisation: utiliser une approche par batch pour réduire les surcoûts
        batch_size = 1000  # Taille maximale du batch pour éviter la surcharge de mémoire
        
        # Diviser les éléments en lots si nécessaire
        for i in range(0, len(elements_a), batch_size):
            batch_a = elements_a[i:i+batch_size]
            
            for j in range(0, len(elements_b), batch_size):
                batch_b = elements_b[j:j+batch_size]
                
                # Trouver les candidats potentiels de clash
                clash_candidates = []
                for elem_a in batch_a:
                    for elem_b in batch_b:
                        # Pré-filtre rapide: vérifier si les boîtes englobantes sont à proximité
                        if self._quick_bbox_proximity(elem_a, elem_b, self.tolerance * 2):
                            clash_candidates.append((elem_a, elem_b))
                
                # Utiliser ThreadPoolExecutor pour paralléliser la vérification approfondie
                with ThreadPoolExecutor() as executor:
                    futures = []
                    
                    for elem_a, elem_b in clash_candidates:
                        futures.append(executor.submit(
                            self._check_detailed_clash,
                            elem_a,
                            elem_b,
                            self.tolerance
                        ))
                    
                    # Collecter les résultats
                    for future in futures:
                        result = future.result()
                        if result:
                            model_clashes.append(result)
        
        return model_clashes
    
    def _quick_bbox_proximity(self, elem_a, elem_b, max_distance):
        """Vérifie rapidement si deux boîtes englobantes sont à proximité l'une de l'autre"""
        # Calculer les distances entre les boîtes englobantes
        min_a = elem_a['bbox_min']
        max_a = elem_a['bbox_max']
        
        min_b = elem_b['bbox_min']
        max_b = elem_b['bbox_max']
        
        # Si les boîtes s'intersectent, elles sont à proximité
        if np.all(min_a <= max_b) and np.all(min_b <= max_a):
            return True
        
        # Sinon, calculer la distance la plus courte entre les boîtes
        # Pour chaque dimension, trouver la distance minimale
        dist = 0
        for i in range(3):  # 3 dimensions (x, y, z)
            if min_a[i] > max_b[i]:
                dist += (min_a[i] - max_b[i]) ** 2
            elif min_b[i] > max_a[i]:
                dist += (min_b[i] - max_a[i]) ** 2
        
        return np.sqrt(dist) <= max_distance
    
    def _check_detailed_clash(self, elem_a, elem_b, tolerance):
        """Vérifie en détail si deux éléments sont en clash"""
        # Pour commencer, vérifions si les boîtes englobantes s'intersectent
        min_a = elem_a['bbox_min'] - tolerance
        max_a = elem_a['bbox_max'] + tolerance
        
        min_b = elem_b['bbox_min'] - tolerance
        max_b = elem_b['bbox_max'] + tolerance
        
        # Vérifier si les boîtes englobantes s'intersectent
        if np.any(min_a > max_b) or np.any(min_b > max_a):
            return None  # Pas d'intersection
        
        # Calculer le volume de l'intersection
        intersection_min = np.maximum(min_a, min_b)
        intersection_max = np.minimum(max_a, max_b)
        
        # Volume de l'intersection
        intersection_volume = np.prod(np.maximum(0, intersection_max - intersection_min))
        
        # Volume relatif par rapport au plus petit élément
        relative_volume = intersection_volume / min(elem_a['volume'], elem_b['volume'])
        
        # Calculer la distance entre les centres
        dist = np.linalg.norm(elem_a['center'] - elem_b['center'])
        
        # Critères de clash:
        # 1. Volume d'intersection significatif OU
        # 2. Distance entre centres inférieure à la tolérance
        if relative_volume > 0.01 or dist < tolerance:
            # Position du clash (point médian de l'intersection)
            position = (intersection_min + intersection_max) / 2
            
            return {
                'element_a': {
                    'guid': elem_a['guid'],
                    'name': elem_a['name'],
                    'type': elem_a['type'],
                    'model': elem_a['model_name']
                },
                'element_b': {
                    'guid': elem_b['guid'],
                    'name': elem_b['name'],
                    'type': elem_b['type'],
                    'model': elem_b['model_name']
                },
                'distance': float(dist),
                'overlap_volume': float(intersection_volume),
                'relative_overlap': float(relative_volume),
                'position': position.tolist(),
                'center_a': elem_a['center'].tolist(),
                'center_b': elem_b['center'].tolist()
            }
        
        return None
        
    def debug_model_info(self):
        """Affiche des informations de débogage sur les modèles chargés"""
        if not self.models:
            print("Aucun modèle n'a été chargé")
            return
            
        print("\n=== Informations sur les modèles chargés ===")
        for i, model in enumerate(self.models):
            print(f"Modèle {i+1}: {model['model_name']}")
            print(f"  Nombre d'éléments: {len(model['elements'])}")
            print(f"  Limites du modèle:")
            print(f"    X: {model['bounds_min'][0]:.2f} à {model['bounds_max'][0]:.2f}")
            print(f"    Y: {model['bounds_min'][1]:.2f} à {model['bounds_max'][1]:.2f}")
            print(f"    Z: {model['bounds_min'][2]:.2f} à {model['bounds_max'][2]:.2f}")
            
            # Calculer la diagonale du modèle
            diagonal = np.linalg.norm(model['bounds_max'] - model['bounds_min'])
            print(f"  Diagonale du modèle: {diagonal:.2f}m")
            
            # Afficher les types d'éléments présents
            element_types = {}
            for elem in model['elements']:
                if elem['type'] not in element_types:
                    element_types[elem['type']] = 0
                element_types[elem['type']] += 1
            
            print("  Types d'éléments:")
            for etype, count in element_types.items():
                print(f"    {etype}: {count}")
            
            print("")