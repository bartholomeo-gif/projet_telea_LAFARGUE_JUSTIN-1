"""Fonctions personnalisées pour le projet de télédétection
Cours : Télédétection Approfondissement - Qualité et fouille de données
Auteur : Justin Lafargue
"""

import numpy as np


def calculate_ari(B03, B05, nodata=-9999):
    """Calcule l'indice ARI (Anthocyanin Reflectance Index).

    ARI = (1/B03 - 1/B05) / (1/B03 + 1/B05)

    Parameters
    ----------
    B03 : ndarray
        Bande verte Sentinel-2 (rows, cols, dates)
    B05 : ndarray
        Bande red-edge Sentinel-2 (rows, cols, dates)
    nodata : float
        Valeur nodata à utiliser

    Returns
    -------
    ari : ndarray
        Série temporelle ARI
    """
    np.seterr(divide="ignore", invalid="ignore")

    ari = (1.0 / B03 - 1.0 / B05) / (1.0 / B03 + 1.0 / B05)
    ari[np.isnan(ari)] = nodata

    return ari


def extract_stats_by_class(image, mask, classes, nodata=-9999):
    """Calcule la moyenne et l'écart-type par classe et par date.

    Parameters
    ----------
    image : ndarray (rows, cols, dates)
        Série temporelle (ex: ARI)
    mask : ndarray (rows, cols)
        Raster des classes
    classes : list
        Liste des labels de classes à analyser
    nodata : float
        Valeur nodata à exclure

    Returns
    -------
    means : ndarray (nb_classes, nb_dates)
    stds : ndarray (nb_classes, nb_dates)
    """
    nb_dates = image.shape[2]
    nb_classes = len(classes)

    means = np.zeros((nb_classes, nb_dates))
    stds = np.zeros((nb_classes, nb_dates))

    for i, cls in enumerate(classes):
        cls_mask = mask == cls
        for d in range(nb_dates):
            values = image[:, :, d][cls_mask]
            values = values[values != nodata]

            if values.size > 0:
                means[i, d] = np.mean(values)
                stds[i, d] = np.std(values)
            else:
                means[i, d] = np.nan
                stds[i, d] = np.nan

    return means, stds


def prepare_training_data(X_image, y_raster):
    """Prépare les données pour une classification supervisée.

    Parameters
    ----------
    X_image : ndarray (rows, cols, n_features)
        Image des variables explicatives (bandes spectrales empilées)
    y_raster : ndarray (rows, cols)
        Raster des strates (classes)

    Returns
    -------
    X : ndarray (n_samples, n_features)
        Matrice d'apprentissage
    y : ndarray (n_samples,)
        Labels
    mask : ndarray bool
        Masque des pixels échantillonnés
    """
    rows, cols, n_features = X_image.shape

    # Pixels étiquetés uniquement (nodata = 0)
    mask = y_raster > 0

    X_flat = X_image.reshape(rows * cols, n_features)
    y_flat = y_raster.flatten()

    X = X_flat[mask.flatten(), :]
    y = y_flat[mask.flatten()]

    return X, y, mask
