from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import os
import tempfile
from .clash_detection import FastClashDetector
import ifcopenshell
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

def _get_ifc_units(ifc_path):
    """Récupère les unités du fichier IFC pour le debug"""
    try:
        model = ifcopenshell.open(ifc_path)
        unit_assignments = model.by_type("IfcUnitAssignment")
        if unit_assignments:
            return str(unit_assignments[0])
        return "no units found"
    except Exception as e:
        return f"error reading units: {str(e)}"

@main.route('/api/fast_clash', methods=['POST'])
@cross_origin()
def fast_clash_detection():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.ifc'):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Paramètres avec valeurs par défaut plus permissives
        tolerance = float(request.form.get('tolerance', 0.1))  # Augmenté à 10cm par défaut
        use_ai = request.form.get('use_ai', 'false').lower() == 'true'  # Désactivé par défaut
        debug = request.form.get('debug', 'false').lower() == 'true'
        
        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ifc') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            logger.info(f"Starting detection with tolerance: {tolerance}, AI: {use_ai}, Debug: {debug}")
            
            detector = FastClashDetector(tolerance=tolerance, use_ai=use_ai, debug=debug)
            elements = detector.load_model(tmp_path)
            logger.info(f"Loaded {len(elements)} elements")
            
            # Debug: afficher quelques éléments
            if debug and elements:
                logger.info("First few elements:")
                for i, elem in enumerate(elements[:3]):
                    logger.info(f"  {i}: {elem['type']} - {elem['guid']}")
                    logger.info(f"     BBox: {elem['bbox_min']} to {elem['bbox_max']}")
                    logger.info(f"     Center: {elem['center']}, Volume: {elem['volume']}")
            
            clashes = detector.detect_clashes(elements)
            logger.info(f"Detection completed with {len(clashes)} clashes")
            
            # Ajout de statistiques de debug
            debug_stats = {}
            if elements:
                import numpy as np
                centers = np.array([e['center'] for e in elements])
                volumes = np.array([e['volume'] for e in elements])
                
                debug_stats = {
                    "center_bounds": {
                        "min": centers.min(axis=0).tolist(),
                        "max": centers.max(axis=0).tolist()
                    },
                    "volume_stats": {
                        "min": float(volumes.min()),
                        "max": float(volumes.max()),
                        "mean": float(volumes.mean())
                    },
                    "element_types": {}
                }
                
                # Comptage des types d'éléments
                for elem in elements:
                    elem_type = elem['type']
                    debug_stats["element_types"][elem_type] = debug_stats["element_types"].get(elem_type, 0) + 1
            
            return jsonify({
                "status": "success",
                "clash_count": len(clashes),
                "clashes": clashes,
                "settings": {
                    "tolerance": tolerance,
                    "ai_filtering": use_ai,
                    "debug": debug
                },
                "model_stats": {
                    "elements_processed": len(elements),
                    "ifc_units": _get_ifc_units(tmp_path)
                },
                "debug_stats": debug_stats if debug else None
            })
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "hint": "Try increasing tolerance value or disabling AI filtering"
        }), 500

@main.route('/api/debug_clash', methods=['POST'])
@cross_origin()
def debug_clash_detection():
    """Route spéciale pour le debug avec paramètres optimisés"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.ifc'):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Paramètres optimisés pour détecter des clashes
        tolerance = 0.5  # 50cm - très permissif
        use_ai = False   # Pas de filtrage
        debug = True     # Debug activé
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ifc') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            logger.info("Starting DEBUG detection with optimized parameters")
            
            detector = FastClashDetector(tolerance=tolerance, use_ai=use_ai, debug=debug)
            elements = detector.load_model(tmp_path)
            clashes = detector.detect_clashes(elements)
            
            return jsonify({
                "status": "success",
                "clash_count": len(clashes),
                "clashes": clashes,
                "message": "Debug mode with optimized parameters",
                "settings": {
                    "tolerance": tolerance,
                    "ai_filtering": use_ai,
                    "debug": debug
                }
            })
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error in debug detection: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500