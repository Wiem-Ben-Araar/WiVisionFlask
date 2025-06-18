from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import ifcopenshell
from .clash_detection import FastClashDetector
import time
import os

main = Blueprint('main', __name__)

@main.route('/api/fast_clash', methods=['POST'])
@cross_origin()
def fast_clash_detection():
    if 'file' not in request.files:
        return jsonify({"error": "No IFC file provided"}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.ifc'):
        return jsonify({"error": "Invalid IFC file"}), 400

    try:
        # Paramètres
        tolerance = float(request.form.get('tolerance', 0.01))
        use_ai = request.form.get('use_ai', 'true').lower() == 'true'
        
        # Sauvegarde temporaire
        temp_path = f"/tmp/{int(time.time())}.ifc"
        file.save(temp_path)
        
        # Détection ultra-rapide
        detector = FastClashDetector(tolerance=tolerance, use_ai=use_ai)
        elements = detector.load_model(temp_path)
        clashes = detector.detect_clashes(elements)
        
        # Nettoyage
        os.remove(temp_path)
        
        return jsonify({
            "status": "success",
            "clash_count": len(clashes),
            "clashes": clashes,
            "performance": {
                "elements_processed": len(elements),
                "ai_filtering": use_ai
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500