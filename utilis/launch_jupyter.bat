@echo off
:: Spécifie le chemin complet vers le fichier "activate" d'Anaconda
call "C:\anaconda_install\Scripts\activate.bat" data_science

:: Lance Jupyter Notebook
jupyter notebook
