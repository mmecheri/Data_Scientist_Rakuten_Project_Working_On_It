::@echo off
:: Désactive l'affichage des commandes exécutées
::cls

:: Spécifie le chemin complet vers Anaconda et active l'environnement "data_science"
call "C:\anaconda_install\Scripts\activate.bat" data_science

:: Affiche un message de confirmation
:: echo ✅ Conda environment 'data_science' activated.
:: echo You can now install packages using pip or conda.

:: Ouvre une nouvelle invite de commande avec l'environnement activé
cmd
