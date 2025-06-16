from flask import Blueprint, request, jsonify, current_app, render_template
from flask_cors import cross_origin
import os
import uuid
import json
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from .clash_detection import FastClashDetector

main = Blueprint('main', __name__)
executor = ThreadPoolExecutor(max_workers=4)

# Autoriser les requêtes cross-origin pour React (CORS)
@main.route('/api/clash/detect', methods=['POST', 'OPTIONS'])
@cross_origin()
def clash_detect():
    if 'models' not in request.files:
        return jsonify(error="No IFC files provided"), 400

    model_files = request.files.getlist('models')
    if len(model_files) < 2:
        return jsonify(error="At least 2 IFC files required"), 400

    # Debug: Afficher les fichiers reçus
    print(f"Received files: {[f.filename for f in model_files]}")

    tolerance = float(request.form.get('tolerance', 0.01))
    session_id = str(uuid.uuid4())
    saved_paths = []

    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        for file in model_files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            session_folder = os.path.join(upload_folder, session_id)
            os.makedirs(session_folder, exist_ok=True)
            file_path = os.path.join(session_folder, filename)
            file.save(file_path)
            saved_paths.append(file_path)
            print(f"Saved file: {file_path}")

        # Démarrer le traitement
        executor.submit(
            process_clash_detection,
            current_app._get_current_object(),  # Passez l'instance réelle de l'application
            session_id, 
            saved_paths, 
            tolerance
        )
        
       
        return jsonify({
            "session_id": session_id,
            "status": "processing",
            "status_url": f"/api/status/{session_id}"  # URL relative
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify(error=str(e)), 500


def process_clash_detection(app, session_id, paths, tolerance):
    """Fonction exécutée en arrière-plan"""
    with app.app_context():
        try:
            print(f"\n=== Début du traitement pour {session_id} ===")
            
            detector = FastClashDetector(tolerance)
            for path in paths:
                print(f"Chargement du modèle: {path}")
                detector.add_model(path)
                print(f"Modèle chargé: {os.path.basename(path)}")
            
            print("Lancement de la détection des clashs...")
            clashes = detector.detect_clashes()
            print(f"Clashs détectés: {len(clashes)}")
            
            # Générer des données pour les visualisations 3D
            clash_visualizations = generate_clash_visualizations(clashes, paths, detector)
            
            report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, 'report.json')
            
            with open(report_path, 'w') as f:
                json.dump({
                    'session_id': session_id,
                    'status': 'completed',
                    'clash_count': len(clashes),
                    'clashes': clashes,
                    'visualizations': clash_visualizations
                }, f, indent=2)
            
            # Générer un rapport HTML
            generate_html_report(session_id, clashes, clash_visualizations, app)
            
            print(f"Rapport sauvegardé: {report_path}")
            print(f"=== Traitement terminé pour {session_id} ===\n")

        except Exception as e:
            import traceback
            error_msg = f"ERREUR: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
            os.makedirs(report_dir, exist_ok=True)
            error_path = os.path.join(report_dir, 'error.json')
            
            with open(error_path, 'w') as f:
                json.dump({
                    'error': error_msg,
                    'status': 'failed'
                }, f)


def generate_clash_visualizations(clashes, model_paths, detector):
    """Génère des données pour visualiser les clashs en 3D"""
    visualizations = []
    
    # Charger tous les modèles IFC une seule fois
    ifc_models = {}
    for path in model_paths:
        model_name = os.path.basename(path)
        ifc_models[model_name] = ifcopenshell.open(path)
    
    # Configurer le générateur de géométrie
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    # Pour chaque clash, générer des données de visualisation
    for i, clash in enumerate(clashes[:100]):  # Limiter à 100 clashs pour éviter des rapports trop lourds
        try:
            # Extraire les informations des éléments en clash
            elem_a_guid = clash['element_a']['guid']
            elem_b_guid = clash['element_b']['guid']
            model_a_name = clash['element_a']['model']
            model_b_name = clash['element_b']['model']
            
            # Récupérer les modèles IFC correspondants
            model_a = ifc_models.get(model_a_name)
            model_b = ifc_models.get(model_b_name)
            
            if not model_a or not model_b:
                continue
                
            # Récupérer les éléments par GUID
            elem_a = model_a.by_guid(elem_a_guid)
            elem_b = model_b.by_guid(elem_b_guid)
            # Extraire les géométries des éléments
            geom_a = detector.get_element_geometry(elem_a)
            geom_b = detector.get_element_geometry(elem_b)
            # Extraire la géométrie simplifiée pour la visualisation
            vis_data = {
                'id': i,
                'clash_id': f"clash_{i}",
                'position': clash['position'],
                'elements': [
                    {
                        'guid': elem_a_guid,
                        'name': clash['element_a']['name'],
                        'model': model_a_name,
                        'type': clash['element_a']['type']
                    },
                    {
                        'guid': elem_b_guid,
                        'name': clash['element_b']['name'],
                        'model': model_b_name,
                        'type': clash['element_b']['type']
                    }
                ],
                'geometries': {
                    'element_a': geom_a,
                    'element_b': geom_b
                },
                'clash_sphere': {
                    'position': clash['position'],
                    'radius': detector.tolerance * 2
                },
                'distance': clash['distance'],
                'overlap_volume': clash.get('overlap_volume', 0)
            }
            
            visualizations.append(vis_data)
            
        except Exception as e:
            print(f"Erreur lors de la génération de la visualisation pour le clash {i}: {str(e)}")
            continue
    
    return visualizations
@main.route('/api/clash/visualization/<session_id>')
@cross_origin()
def get_clash_visualization(session_id):
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.json')
    
    if not os.path.exists(report_path):
        return jsonify(error="Visual data not found"), 404
        
    with open(report_path) as f:
        data = json.load(f)
        return jsonify(data.get('visualizations', []))

def generate_html_report(session_id, clashes, visualizations, app):
    """Génère un rapport HTML avec des visualisations"""
    report_dir = os.path.join(app.config['REPORTS_FOLDER'], session_id)
    html_path = os.path.join(report_dir, 'report.html')
    
    # Regrouper les clashs par type d'élément pour analyse
    clash_by_type = {}
    for clash in clashes:
        type_a = clash['element_a']['type']
        type_b = clash['element_b']['type']
        key = f"{type_a} vs {type_b}"
        
        if key not in clash_by_type:
            clash_by_type[key] = []
            
        clash_by_type[key].append(clash)
    
    # Générer un rapport HTML simple
    with open(html_path, 'w') as f:
        f.write('''
        <script>

const viewerScript = document.createElement('script');
viewerScript.src = '/static/clash-viewer.js';
document.head.appendChild(viewerScript);
</script>
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport de détection de clash</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .summary { background-color: #e7f3fe; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                h1, h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Rapport de détection de clash</h1>
            
            <div class="summary">
                <h2>Résumé</h2>
                <p>Session ID: ''' + session_id + '''</p>
                <p>Nombre total de clashs: ''' + str(len(clashes)) + '''</p>
            </div>
            
            <div class="section">
                <h2>Clashs par type d'élément</h2>
                <table>
                    <tr>
                        <th>Types d'éléments</th>
                        <th>Nombre de clashs</th>
                    </tr>
        ''')
        
        # Ajouter les statistiques par type
        for type_key, type_clashes in clash_by_type.items():
            f.write(f'''
                    <tr>
                        <td>{type_key}</td>
                        <td>{len(type_clashes)}</td>
                    </tr>
            ''')
            
        f.write('''
                </table>
            </div>
            
            <div class="section">
                <h2>Liste des clashs</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Élément A</th>
                        <th>Élément B</th>
                        <th>Distance</th>
                        <th>Volume de chevauchement</th>
                    </tr>
        ''')
        
        # Ajouter tous les clashs
        for i, clash in enumerate(clashes):
            elem_a = clash['element_a']['name']
            elem_b = clash['element_b']['name']
            distance = clash['distance']
            overlap = clash.get('overlap_volume', 'N/A')
            
            f.write(f'''
                    <tr>
                        <td>{i + 1}</td>
                        <td>{elem_a} ({clash['element_a']['type']})</td>
                        <td>{elem_b} ({clash['element_b']['type']})</td>
                        <td>{distance:.4f} m</td>
                        <td>{overlap if overlap == 'N/A' else f"{overlap:.6f} m³"}</td>
                    </tr>
            ''')
            
        f.write('''
                </table>
            </div>
        </body>
        </html>
        ''')
    
    return html_path


@main.route('/api/status/<session_id>')
@cross_origin()  # Ajouter CORS ici aussi
def check_status(session_id):
    """Endpoint de vérification du statut"""
    report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
    
    if not os.path.exists(report_dir):
        return jsonify({"status": "processing"}), 200
        
    if os.path.exists(os.path.join(report_dir, 'report.json')):
        with open(os.path.join(report_dir, 'report.json')) as f:
            data = json.load(f)
            return jsonify(data)
    
    if os.path.exists(os.path.join(report_dir, 'error.json')):
        with open(os.path.join(report_dir, 'error.json')) as f:
            return jsonify(json.load(f)), 500
            
    return jsonify({
        "status": "processing",
        "message": "L'analyse est en cours..."
    })

@main.route('/api/report/<session_id>')
@cross_origin()  # Ajouter CORS ici aussi
def get_report(session_id):
    """Endpoint pour récupérer le rapport complet"""
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.json')
    
    if not os.path.exists(report_path):
        return jsonify(error="Rapport non trouvé"), 404
        
    with open(report_path) as f:
        return jsonify(json.load(f))

@main.route('/api/report/html/<session_id>')
@cross_origin()  # Ajouter CORS pour l'accès au rapport HTML
def get_html_report(session_id):
    """Endpoint pour accéder au rapport HTML"""
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.html')
    
    if not os.path.exists(report_path):
        return jsonify(error="Rapport HTML non trouvé"), 404
        
    with open(report_path, 'r') as f:
        html_content = f.read()
    
    return html_content

# Fonctionnalités optionnelles
@main.route('/api/clash/history')
@cross_origin()  # Ajouter CORS ici aussi
def detection_history():
    """Liste des analyses récentes"""
    reports_dir = current_app.config['REPORTS_FOLDER']
    sessions = []
    
    for entry in os.listdir(reports_dir):
        if os.path.isdir(os.path.join(reports_dir, entry)):
            sessions.append({
                'session_id': entry,
                'created_at': os.path.getctime(os.path.join(reports_dir, entry))
            })
    
    return jsonify(sorted(sessions, key=lambda x: x['created_at'], reverse=True))
from flask import Blueprint, request, jsonify, current_app, render_template
from flask_cors import cross_origin
import os
import uuid
import json
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from .clash_detection import FastClashDetector

main = Blueprint('main', __name__)
executor = ThreadPoolExecutor(max_workers=4)

# Autoriser les requêtes cross-origin pour React (CORS)
@main.route('/api/clash/detect', methods=['POST', 'OPTIONS'])
@cross_origin()
def clash_detect():
    if 'models' not in request.files:
        return jsonify(error="No IFC files provided"), 400

    model_files = request.files.getlist('models')
    if len(model_files) < 2:
        return jsonify(error="At least 2 IFC files required"), 400

    # Debug: Afficher les fichiers reçus
    print(f"Received files: {[f.filename for f in model_files]}")

    tolerance = float(request.form.get('tolerance', 0.01))
    session_id = str(uuid.uuid4())
    saved_paths = []

    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        for file in model_files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            session_folder = os.path.join(upload_folder, session_id)
            os.makedirs(session_folder, exist_ok=True)
            file_path = os.path.join(session_folder, filename)
            file.save(file_path)
            saved_paths.append(file_path)
            print(f"Saved file: {file_path}")

        # Démarrer le traitement
        executor.submit(
            process_clash_detection,
            current_app._get_current_object(),  # Passez l'instance réelle de l'application
            session_id, 
            saved_paths, 
            tolerance
        )
        
       
        return jsonify({
            "session_id": session_id,
            "status": "processing",
            "status_url": f"/api/status/{session_id}"  # URL relative
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify(error=str(e)), 500


def process_clash_detection(app, session_id, paths, tolerance):
    """Fonction exécutée en arrière-plan"""
    with app.app_context():
        try:
            print(f"\n=== Début du traitement pour {session_id} ===")
            
            detector = FastClashDetector(tolerance)
            for path in paths:
                print(f"Chargement du modèle: {path}")
                detector.add_model(path)
                print(f"Modèle chargé: {os.path.basename(path)}")
            
            print("Lancement de la détection des clashs...")
            clashes = detector.detect_clashes()
            print(f"Clashs détectés: {len(clashes)}")
            
            # Générer des données pour les visualisations 3D
            clash_visualizations = generate_clash_visualizations(clashes, paths, detector)
            
            report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, 'report.json')
            
            with open(report_path, 'w') as f:
                json.dump({
                    'session_id': session_id,
                    'status': 'completed',
                    'clash_count': len(clashes),
                    'clashes': clashes,
                    'visualizations': clash_visualizations
                }, f, indent=2)
            
            # Générer un rapport HTML
            generate_html_report(session_id, clashes, clash_visualizations, app)
            
            print(f"Rapport sauvegardé: {report_path}")
            print(f"=== Traitement terminé pour {session_id} ===\n")

        except Exception as e:
            import traceback
            error_msg = f"ERREUR: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
            os.makedirs(report_dir, exist_ok=True)
            error_path = os.path.join(report_dir, 'error.json')
            
            with open(error_path, 'w') as f:
                json.dump({
                    'error': error_msg,
                    'status': 'failed'
                }, f)


def generate_clash_visualizations(clashes, model_paths, detector):
    """Génère des données pour visualiser les clashs en 3D"""
    visualizations = []
    
    # Charger tous les modèles IFC une seule fois
    ifc_models = {}
    for path in model_paths:
        model_name = os.path.basename(path)
        ifc_models[model_name] = ifcopenshell.open(path)
    
    # Configurer le générateur de géométrie
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    # Pour chaque clash, générer des données de visualisation
    for i, clash in enumerate(clashes[:100]):  # Limiter à 100 clashs pour éviter des rapports trop lourds
        try:
            # Extraire les informations des éléments en clash
            elem_a_guid = clash['element_a']['guid']
            elem_b_guid = clash['element_b']['guid']
            model_a_name = clash['element_a']['model']
            model_b_name = clash['element_b']['model']
            
            # Récupérer les modèles IFC correspondants
            model_a = ifc_models.get(model_a_name)
            model_b = ifc_models.get(model_b_name)
            
            if not model_a or not model_b:
                continue
                
            # Récupérer les éléments par GUID
            elem_a = model_a.by_guid(elem_a_guid)
            elem_b = model_b.by_guid(elem_b_guid)
            # Extraire les géométries des éléments
            geom_a = detector.get_element_geometry(elem_a)
            geom_b = detector.get_element_geometry(elem_b)
            # Extraire la géométrie simplifiée pour la visualisation
            vis_data = {
                'id': i,
                'clash_id': f"clash_{i}",
                'position': clash['position'],
                'elements': [
                    {
                        'guid': elem_a_guid,
                        'name': clash['element_a']['name'],
                        'model': model_a_name,
                        'type': clash['element_a']['type']
                    },
                    {
                        'guid': elem_b_guid,
                        'name': clash['element_b']['name'],
                        'model': model_b_name,
                        'type': clash['element_b']['type']
                    }
                ],
                'geometries': {
                    'element_a': geom_a,
                    'element_b': geom_b
                },
                'clash_sphere': {
                    'position': clash['position'],
                    'radius': detector.tolerance * 2
                },
                'distance': clash['distance'],
                'overlap_volume': clash.get('overlap_volume', 0)
            }
            
            visualizations.append(vis_data)
            
        except Exception as e:
            print(f"Erreur lors de la génération de la visualisation pour le clash {i}: {str(e)}")
            continue
    
    return visualizations
@main.route('/api/clash/visualization/<session_id>')
@cross_origin()
def get_clash_visualization(session_id):
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.json')
    
    if not os.path.exists(report_path):
        return jsonify(error="Visual data not found"), 404
        
    with open(report_path) as f:
        data = json.load(f)
        return jsonify(data.get('visualizations', []))

def generate_html_report(session_id, clashes, visualizations, app):
    """Génère un rapport HTML avec des visualisations"""
    report_dir = os.path.join(app.config['REPORTS_FOLDER'], session_id)
    html_path = os.path.join(report_dir, 'report.html')
    
    # Regrouper les clashs par type d'élément pour analyse
    clash_by_type = {}
    for clash in clashes:
        type_a = clash['element_a']['type']
        type_b = clash['element_b']['type']
        key = f"{type_a} vs {type_b}"
        
        if key not in clash_by_type:
            clash_by_type[key] = []
            
        clash_by_type[key].append(clash)
    
    # Générer un rapport HTML simple
    with open(html_path, 'w') as f:
        f.write('''
        <script>

const viewerScript = document.createElement('script');
viewerScript.src = '/static/clash-viewer.js';
document.head.appendChild(viewerScript);
</script>
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport de détection de clash</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .summary { background-color: #e7f3fe; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                h1, h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Rapport de détection de clash</h1>
            
            <div class="summary">
                <h2>Résumé</h2>
                <p>Session ID: ''' + session_id + '''</p>
                <p>Nombre total de clashs: ''' + str(len(clashes)) + '''</p>
            </div>
            
            <div class="section">
                <h2>Clashs par type d'élément</h2>
                <table>
                    <tr>
                        <th>Types d'éléments</th>
                        <th>Nombre de clashs</th>
                    </tr>
        ''')
        
        # Ajouter les statistiques par type
        for type_key, type_clashes in clash_by_type.items():
            f.write(f'''
                    <tr>
                        <td>{type_key}</td>
                        <td>{len(type_clashes)}</td>
                    </tr>
            ''')
            
        f.write('''
                </table>
            </div>
            
            <div class="section">
                <h2>Liste des clashs</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Élément A</th>
                        <th>Élément B</th>
                        <th>Distance</th>
                        <th>Volume de chevauchement</th>
                    </tr>
        ''')
        
        # Ajouter tous les clashs
        for i, clash in enumerate(clashes):
            elem_a = clash['element_a']['name']
            elem_b = clash['element_b']['name']
            distance = clash['distance']
            overlap = clash.get('overlap_volume', 'N/A')
            
            f.write(f'''
                    <tr>
                        <td>{i + 1}</td>
                        <td>{elem_a} ({clash['element_a']['type']})</td>
                        <td>{elem_b} ({clash['element_b']['type']})</td>
                        <td>{distance:.4f} m</td>
                        <td>{overlap if overlap == 'N/A' else f"{overlap:.6f} m³"}</td>
                    </tr>
            ''')
            
        f.write('''
                </table>
            </div>
        </body>
        </html>
        ''')
    
    return html_path


@main.route('/api/status/<session_id>')
@cross_origin()  # Ajouter CORS ici aussi
def check_status(session_id):
    """Endpoint de vérification du statut"""
    report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
    
    if not os.path.exists(report_dir):
        return jsonify({"status": "processing"}), 200
        
    if os.path.exists(os.path.join(report_dir, 'report.json')):
        with open(os.path.join(report_dir, 'report.json')) as f:
            data = json.load(f)
            return jsonify(data)
    
    if os.path.exists(os.path.join(report_dir, 'error.json')):
        with open(os.path.join(report_dir, 'error.json')) as f:
            return jsonify(json.load(f)), 500
            
    return jsonify({
        "status": "processing",
        "message": "L'analyse est en cours..."
    })

@main.route('/api/report/<session_id>')
@cross_origin()  # Ajouter CORS ici aussi
def get_report(session_id):
    """Endpoint pour récupérer le rapport complet"""
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.json')
    
    if not os.path.exists(report_path):
        return jsonify(error="Rapport non trouvé"), 404
        
    with open(report_path) as f:
        return jsonify(json.load(f))

@main.route('/api/report/html/<session_id>')
@cross_origin()  # Ajouter CORS pour l'accès au rapport HTML
def get_html_report(session_id):
    """Endpoint pour accéder au rapport HTML"""
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.html')
    
    if not os.path.exists(report_path):
        return jsonify(error="Rapport HTML non trouvé"), 404
        
    with open(report_path, 'r') as f:
        html_content = f.read()
    
    return html_content

# Fonctionnalités optionnelles
@main.route('/api/clash/history')
@cross_origin()  # Ajouter CORS ici aussi
def detection_history():
    """Liste des analyses récentes"""
    reports_dir = current_app.config['REPORTS_FOLDER']
    sessions = []
    
    for entry in os.listdir(reports_dir):
        if os.path.isdir(os.path.join(reports_dir, entry)):
            sessions.append({
                'session_id': entry,
                'created_at': os.path.getctime(os.path.join(reports_dir, entry))
            })
    
    return jsonify(sorted(sessions, key=lambda x: x['created_at'], reverse=True))