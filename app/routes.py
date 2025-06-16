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

def generate_html_report(session_id, clashes, visualizations, app, intra_model=False, model_name=None):
    """Génère un rapport HTML avec des visualisations 3D"""
    report_dir = os.path.join(app.config['REPORTS_FOLDER'], session_id)
    html_path = os.path.join(report_dir, 'report.html')
    
    # Titre en fonction du type de détection
    report_title = "Rapport de détection intra-modèle" if intra_model else "Rapport de détection de clash"
    
    # Générer un rapport HTML
    with open(html_path, 'w') as f:
        # En-tête HTML avec intégration de Three.js
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e7f3fe; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                h1, h2 {{ color: #333; }}
                #visualization {{ width: 100%; height: 600px; border: 1px solid #ccc; margin-top: 20px; }}
                .clash-details {{ display: flex; }}
                .clash-table {{ flex: 1; }}
                .clash-viewer {{ flex: 1; }}
                .controls {{ margin: 10px 0; }}
            </style>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://threejs.org/examples/js/controls/OrbitControls.js"></script>
        </head>
        <body>
            <h1>{report_title}</h1>
        ''')
        
        # Section résumé
        if intra_model:
            f.write(f'''
            <div class="summary">
                <h2>Résumé</h2>
                <p>Session ID: {session_id}</p>
                <p>Modèle analysé: {model_name}</p>
                <p>Nombre total de clashs: {len(clashes)}</p>
            </div>
            ''')
        else:
            f.write(f'''
            <div class="summary">
                <h2>Résumé</h2>
                <p>Session ID: {session_id}</p>
                <p>Nombre total de clashs: {len(clashes)}</p>
            </div>
            ''')
            
        f.write(f'''
            <div class="clash-details">
                <div class="clash-table">
                    <h2>Liste des clashs</h2>
                    <div class="controls">
                        <button onclick="showClash(0)">Premier clash</button>
                        <button onclick="nextClash()">Suivant</button>
                        <button onclick="prevClash()">Précédent</button>
                        <span id="clash-counter">1/{len(clashes)}</span>
                    </div>
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
                        <tr id="clash-row-{i}" onclick="showClash({i})">
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
                <div class="clash-viewer">
                    <h2>Visualisation 3D du clash</h2>
                    <div id="visualization"></div>
                </div>
            </div>
            
            <script>
                // Données des clashs
                const clashes = ''' + json.dumps(visualizations) + ''';
                let currentClashIndex = 0;
                let scene, camera, renderer, controls;
                
                // Initialisation de Three.js
                function initThreeJS() {
                    // Créer la scène
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(0xf0f0f0);
                    
                    // Créer la caméra
                    camera = new THREE.PerspectiveCamera(75, 
                        document.getElementById('visualization').clientWidth / 
                        document.getElementById('visualization').clientHeight, 
                        0.1, 1000);
                    camera.position.z = 5;
                    
                    // Créer le renderer
                    renderer = new THREE.WebGLRenderer({ antialias: true });
                    renderer.setSize(
                        document.getElementById('visualization').clientWidth,
                        document.getElementById('visualization').clientHeight
                    );
                    document.getElementById('visualization').appendChild(renderer.domElement);
                    
                    // Ajouter des contrôles orbitaux
                    controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    
                    // Ajouter un éclairage
                    const ambientLight = new THREE.AmbientLight(0x404040);
                    scene.add(ambientLight);
                    
                    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                    directionalLight.position.set(1, 1, 1).normalize();
                    scene.add(directionalLight);
                    
                    // Afficher le premier clash
                    showClash(0);
                    
                    // Animer la scène
                    function animate() {
                        requestAnimationFrame(animate);
                        controls.update();
                        renderer.render(scene, camera);
                    }
                    animate();
                    
                    // Gérer le redimensionnement
                    window.addEventListener('resize', () => {
                        camera.aspect = document.getElementById('visualization').clientWidth / 
                                        document.getElementById('visualization').clientHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(
                            document.getElementById('visualization').clientWidth,
                            document.getElementById('visualization').clientHeight
                        );
                    });
                }
                
                // Afficher un clash spécifique
                function showClash(index) {
                    // Mettre à jour l'index courant
                    currentClashIndex = index;
                    document.getElementById('clash-counter').textContent = `${index + 1}/${clashes.length}`;
                    
                    // Mettre en surbrillance la ligne sélectionnée
                    document.querySelectorAll('tr[id^="clash-row-"]').forEach(row => {
                        row.style.backgroundColor = '';
                    });
                    document.getElementById(`clash-row-${index}`).style.backgroundColor = '#e0f7fa';
                    
                    // Nettoyer la scène
                    while(scene.children.length > 2) {
                        scene.remove(scene.children[2]);
                    }
                    
                    // Récupérer les données du clash
                    const clash = clashes[index];
                    
                    // Ajouter l'élément A (rouge)
                    const geometryA = new THREE.BufferGeometry();
                    const verticesA = new Float32Array(clash.geometries.element_a.vertices.flat());
                    geometryA.setAttribute('position', new THREE.BufferAttribute(verticesA, 3));
                    const meshA = new THREE.Mesh(
                        geometryA,
                        new THREE.MeshPhongMaterial({ 
                            color: 0xff0000, 
                            transparent: true,
                            opacity: 0.7 
                        })
                    );
                    scene.add(meshA);
                    
                    // Ajouter l'élément B (bleu)
                    const geometryB = new THREE.BufferGeometry();
                    const verticesB = new Float32Array(clash.geometries.element_b.vertices.flat());
                    geometryB.setAttribute('position', new THREE.BufferAttribute(verticesB, 3));
                    const meshB = new THREE.Mesh(
                        geometryB,
                        new THREE.MeshPhongMaterial({ 
                            color: 0x0000ff, 
                            transparent: true,
                            opacity: 0.7 
                        })
                    );
                    scene.add(meshB);
                    
                    // Ajouter la sphère de clash (vert)
                    const sphereGeometry = new THREE.SphereGeometry(clash.clash_sphere.radius, 32, 32);
                    const sphereMaterial = new THREE.MeshBasicMaterial({ 
                        color: 0x00ff00, 
                        wireframe: true 
                    });
                    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
                    sphere.position.set(
                        clash.clash_sphere.position[0],
                        clash.clash_sphere.position[1],
                        clash.clash_sphere.position[2]
                    );
                    scene.add(sphere);
                    
                    // Centrer la caméra sur le clash
                    const bbox = new THREE.Box3().setFromObject(sphere);
                    const center = bbox.getCenter(new THREE.Vector3());
                    const size = bbox.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const fov = camera.fov * (Math.PI / 180);
                    let cameraZ = Math.abs(maxDim / (2 * Math.tan(fov / 2)));
                    cameraZ *= 2; // Ajustement pour voir l'ensemble
                    
                    camera.position.copy(center);
                    camera.position.z += cameraZ;
                    camera.lookAt(center);
                    controls.update();
                }
                
                // Clash suivant
                function nextClash() {
                    const nextIndex = (currentClashIndex + 1) % clashes.length;
                    showClash(nextIndex);
                }
                
                // Clash précédent
                function prevClash() {
                    const prevIndex = (currentClashIndex - 1 + clashes.length) % clashes.length;
                    showClash(prevIndex);
                }
                
                // Initialiser Three.js quand la page est chargée
                window.addEventListener('load', initThreeJS);
            </script>
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
@cross_origin()
def get_html_report(session_id):
    """Endpoint pour accéder au rapport HTML"""
    report_path = os.path.join(current_app.config['REPORTS_FOLDER'], session_id, 'report.html')
    
    if not os.path.exists(report_path):
        return jsonify(error="Rapport HTML non trouvé"), 404
        
    # Renvoyer le rapport HTML directement
    return send_file(report_path)

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
@main.route('/api/clash/detect_intra', methods=['POST', 'OPTIONS'])
@cross_origin()
def intra_clash_detect():
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    
    # Debug: Print all request data
    print("=== DEBUG REQUEST INFO ===")
    print(f"Request method: {request.method}")
    print(f"Request files: {list(request.files.keys())}")
    print(f"Request form: {dict(request.form)}")
    print(f"Content-Type: {request.content_type}")
    print("========================")
    
    # Check if file is present
    if 'model' not in request.files:
        print("ERROR: 'model' key not found in request.files")
        print(f"Available keys: {list(request.files.keys())}")
        return jsonify(error="No IFC file provided - expected 'model' field"), 400

    model_file = request.files['model']
    
    # More detailed file validation
    if not model_file:
        print("ERROR: model_file is None or empty")
        return jsonify(error="Invalid file object"), 400
        
    if model_file.filename == '' or model_file.filename is None:
        print(f"ERROR: Invalid filename: '{model_file.filename}'")
        return jsonify(error="Invalid filename - file must have a name"), 400
    
    # Check file extension
    if not model_file.filename.lower().endswith('.ifc'):
        print(f"ERROR: Invalid file extension for: {model_file.filename}")
        return jsonify(error="File must be an IFC file (.ifc extension)"), 400
    
    print(f"File received: {model_file.filename}, size: {model_file.content_length}")

    try:
        tolerance = float(request.form.get('tolerance', 0.01))
    except (ValueError, TypeError):
        print(f"ERROR: Invalid tolerance value: {request.form.get('tolerance')}")
        return jsonify(error="Invalid tolerance value - must be a number"), 400
    
    use_ai = request.form.get('use_ai', 'true').lower() == 'true'
    session_id = str(uuid.uuid4())
    
    print(f"Parameters: tolerance={tolerance}, use_ai={use_ai}, session_id={session_id}")

    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(model_file.filename)
        if not filename:  # secure_filename might return empty string
            filename = f"model_{session_id}.ifc"
            
        session_folder = os.path.join(upload_folder, session_id)
        os.makedirs(session_folder, exist_ok=True)
        file_path = os.path.join(session_folder, filename)
        
        # Save the file
        model_file.save(file_path)
        
        # Verify file was saved and has content
        if not os.path.exists(file_path):
            raise Exception("File was not saved successfully")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise Exception("Saved file is empty")
            
        print(f"File saved successfully: {file_path} ({file_size} bytes)")

        # Test if it's a valid IFC file by trying to open it
        try:
            import ifcopenshell
            test_model = ifcopenshell.open(file_path)
            print(f"IFC file validation successful: {len(test_model.by_type('IfcElement'))} elements found")
        except Exception as ifc_error:
            print(f"ERROR: Invalid IFC file: {str(ifc_error)}")
            return jsonify(error=f"Invalid IFC file: {str(ifc_error)}"), 400

        # Démarrer le traitement en arrière-plan
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
            "status_url": f"/api/status/{session_id}",
            "file_info": {
                "filename": filename,
                "size": file_size
            }
        })

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify(error=str(e)), 500


def process_intra_clash_detection(app, session_id, path, tolerance, use_ai):
    """Fonction exécutée en arrière-plan pour la détection intra-modèle"""
    with app.app_context():
        try:
            print(f"\n=== Début du traitement intra-modèle pour {session_id} ===")
            
            detector = FastClashDetector(tolerance, debug_mode=True)
            model_info = detector.add_model(path)
            
            if model_info is None:
                raise ValueError("Le modèle n'a pas pu être chargé")
            
            print("Lancement de la détection intra-modèle...")
            clashes = detector.detect_intra_model_clashes(model_info)
            print(f"Clashs intra-modèle détectés: {len(clashes)}")
            
            # Générer des données pour les visualisations 3D
            clash_visualizations = generate_intra_clash_visualizations(clashes, path, detector)
            
            report_dir = os.path.join(current_app.config['REPORTS_FOLDER'], session_id)
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, 'report.json')
            
            with open(report_path, 'w') as f:
                json.dump({
                    'session_id': session_id,
                    'status': 'completed',
                    'clash_count': len(clashes),
                    'clashes': clashes,
                    'visualizations': clash_visualizations,
                    'model_name': model_info['model_name'],
                    'element_count': len(model_info['elements']),
                    'clashing_element_count': len(set(c['element_a']['guid'] for c in clashes) | set(c['element_b']['guid'] for c in clashes)),
                    'ai_used': use_ai
                }, f, indent=2)
            
            # Générer un rapport HTML
            generate_html_report(session_id, clashes, clash_visualizations, app, intra_model=True, model_name=model_info['model_name'])
            
            print(f"Rapport sauvegardé: {report_path}")
            print(f"=== Traitement intra-modèle terminé pour {session_id} ===\n")

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

def generate_intra_clash_visualizations(clashes, model_path, detector):
    """Génère des données pour visualiser les clashs intra-modèle en 3D"""
    visualizations = []
    
    # Charger le modèle IFC
    ifc_model = ifcopenshell.open(model_path)
    model_name = os.path.basename(model_path)
    
    # Configurer le générateur de géométrie
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    # Pour chaque clash, générer des données de visualisation
    for i, clash in enumerate(clashes[:100]):  # Limiter à 100 clashs
        try:
            # Extraire les informations des éléments en clash
            elem_a_guid = clash['element_a']['guid']
            elem_b_guid = clash['element_b']['guid']
            
            # Récupérer les éléments par GUID
            elem_a = ifc_model.by_guid(elem_a_guid)
            elem_b = ifc_model.by_guid(elem_b_guid)
            
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
                        'model': model_name,
                        'type': clash['element_a']['type']
                    },
                    {
                        'guid': elem_b_guid,
                        'name': clash['element_b']['name'],
                        'model': model_name,
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