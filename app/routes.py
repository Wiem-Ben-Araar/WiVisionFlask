from flask import Blueprint, request, jsonify, current_app, send_file
from flask_cors import cross_origin
import os
import uuid
import json
import time
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from .clash_detection import FastIntraModelClashDetector

main = Blueprint('main', __name__)
executor = ThreadPoolExecutor(max_workers=4)

@main.route('/api/intra_clash/detect', methods=['POST'])
@cross_origin()
def intra_clash_detect():
    if 'model' not in request.files:
        return jsonify(error="No IFC file provided"), 400

    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify(error="Empty file"), 400

    try:
        tolerance = float(request.form.get('tolerance', 0.01))
        use_ai = request.form.get('use_ai', 'true').lower() == 'true'
    except ValueError:
        return jsonify(error="Invalid tolerance value"), 400

    session_id = str(uuid.uuid4())
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    try:
        # Sauvegarde du fichier
        filename = secure_filename(model_file.filename)
        session_folder = os.path.join(upload_folder, session_id)
        os.makedirs(session_folder, exist_ok=True)
        file_path = os.path.join(session_folder, filename)
        model_file.save(file_path)

        # Validation rapide du fichier IFC
        try:
            ifc = ifcopenshell.open(file_path)
            if not ifc.by_type("IfcElement"):
                return jsonify(error="No elements found in IFC file"), 400
        except Exception as e:
            return jsonify(error=f"Invalid IFC file: {str(e)}"), 400

        # Lancement du traitement en arrière-plan
        executor.submit(
            process_intra_clash_detection,
            current_app._get_current_object(),
            session_id,
            file_path,
            tolerance,
            use_ai
        )

        return jsonify({
            "session_id": session_id,
            "status": "processing",
            "status_url": f"/api/intra_clash/status/{session_id}",
            "report_url": f"/api/intra_clash/report/{session_id}"
        })

    except Exception as e:
        return jsonify(error=str(e)), 500

def process_intra_clash_detection(app, session_id, ifc_path, tolerance, use_ai):
    with app.app_context():
        try:
            print(f"Starting detection for {session_id}")
            detector = FastIntraModelClashDetector(tolerance, use_ai=use_ai)
            
            if not detector.load_model(ifc_path):
                raise ValueError("Failed to load model")
            
            clashes = detector.detect_clashes()
            model_name = os.path.basename(ifc_path)

            # Génération des visualisations
            visualizations = []
            for i, clash in enumerate(clashes[:100]):  # Limite à 100 pour la performance
                try:
                    geom_a = detector.get_element_geometry(clash['element_a']['guid'])
                    geom_b = detector.get_element_geometry(clash['element_b']['guid'])
                    
                    if geom_a and geom_b:
                        visualizations.append({
                            'id': i,
                            'position': clash['position'],
                            'elements': [
                                {**clash['element_a'], 'model': model_name},
                                {**clash['element_b'], 'model': model_name}
                            ],
                            'geometries': {
                                'element_a': geom_a,
                                'element_b': geom_b
                            },
                            'distance': clash['distance'],
                            'overlap_volume': clash['overlap_volume']
                        })
                except Exception as e:
                    print(f"Visualization error for clash {i}: {str(e)}")
                    continue

            # Sauvegarde du rapport
            report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
            os.makedirs(report_dir, exist_ok=True)
            
            report_data = {
                'session_id': session_id,
                'status': 'completed',
                'model_name': model_name,
                'clash_count': len(clashes),
                'clashes': clashes,
                'visualizations': visualizations,
                'timestamp': time.time()
            }
            
            with open(os.path.join(report_dir, 'report.json'), 'w') as f:
                json.dump(report_data, f, indent=2)
                
            print(f"Detection completed for {session_id} with {len(clashes)} clashes")

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
            os.makedirs(report_dir, exist_ok=True)
            
            with open(os.path.join(report_dir, 'error.json'), 'w') as f:
                json.dump({
                    'error': error_msg,
                    'status': 'failed'
                }, f)

@main.route('/api/intra_clash/status/<session_id>')
@cross_origin()
def intra_clash_status(session_id):
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.json')
    error_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'error.json')
    
    if os.path.exists(error_path):
        with open(error_path) as f:
            return jsonify(json.load(f)), 500
            
    if os.path.exists(report_path):
        with open(report_path) as f:
            return jsonify(json.load(f))
    
    return jsonify({
        "status": "processing",
        "message": "Detection in progress"
    })

@main.route('/api/intra_clash/report/<session_id>')
@cross_origin()
def intra_clash_report(session_id):
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.json')
    
    if not os.path.exists(report_path):
        return jsonify(error="Report not found"), 404
        
    return send_file(report_path, mimetype='application/json')

@main.route('/api/intra_clash/visualize/<session_id>')
@cross_origin()
def intra_clash_visualize(session_id):
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.json')
    
    if not os.path.exists(report_path):
        return jsonify(error="Report not found"), 404
        
    with open(report_path) as f:
        report_data = json.load(f)
        
    return jsonify({
        'clashes': report_data.get('clashes', []),
        'visualizations': report_data.get('visualizations', [])
    })