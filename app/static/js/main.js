// Fonctions JavaScript pour l'application IFC Clash Detection

document.addEventListener('DOMContentLoaded', function() {
    // Initialisation des tooltips Bootstrap si présents
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Gestionnaire pour la validation des fichiers uploadés
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', validateFileType);
    });
    
    // Gestion des messages flash avec disparition automatique
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeButton = alert.querySelector('.btn-close');
            if (closeButton) {
                closeButton.click();
            }
        }, 5000); // Disparaître après 5 secondes
    });
    
    // Configuration des visualiseurs d'images si présents
    setupImageViewers();
});

// Fonction pour valider le type de fichier IFC
function validateFileType(event) {
    const file = event.target.files[0];
    
    if (file) {
        const fileName = file.name;
        const fileExt = fileName.split('.').pop().toLowerCase();
        
        if (fileExt !== 'ifc') {
            alert('Erreur: Seuls les fichiers .ifc sont acceptés.');
            event.target.value = ''; // Effacer la sélection
        }
    }
}

// Configure les visualiseurs d'images pour les clashes
function setupImageViewers() {
    const clashImages = document.querySelectorAll('.card-img-bottom img');
    
    clashImages.forEach(img => {
        img.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Optionellement, implémentez ici un visualiseur d'image
            // ou laissez le comportement par défaut (ouverture dans un nouvel onglet)
        });
    });
}

// Fonction pour filtrer les clashes sur la page de rapport
function filterClashes(searchTerm) {
    const clashCards = document.querySelectorAll('.col-md-6.mb-4');
    
    searchTerm = searchTerm.toLowerCase();
    
    clashCards.forEach(card => {
        const cardText = card.textContent.toLowerCase();
        if (cardText.includes(searchTerm)) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}