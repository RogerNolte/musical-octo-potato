pythonimport numpy as np

def generiere_4d_resonator_mit_sonnenlicht(anzahl_schalen=4, punkte_pro_schale=1000, t=0.0, 
                                          sonnen_vektor=np.array([1.0, 1.0, 1.0])):
    """
    Erweitert das 4D-System um ein paralleles Sonnenlichtfeld.
    Berechnet die Lichtintensität für jeden Punkt basierend auf dem Einfallswinkel.
    """
    phi = 1.61803398875
    w_gold = np.radians(137.51)
    basis_werte = np.array([0.0, 5.360, 5.175, -0.603, -1.519, 1.519, 0.603, -5.175, -5.360])
    
    # Sonnenvektor normalisieren
    sonne_normiert = sonnen_vektor / np.linalg.norm(sonnen_vektor)
    
    alle_vertices = []
    alle_faces = []
    alle_lichtwerte = []
    vertex_offset = 1
    
    # Atem-Effekt des Radius
    atem_faktor = 1.0 + 0.15 * np.sin(t * 2.0)
    kugel_radius = 10.0 * atem_faktor
    
    for schale_id in range(anzahl_schalen):
        innen_radius = 1.0 + (schale_id * 2.0)
        aussen_radius = innen_radius + 1.8
        
        n = np.arange(punkte_pro_schale) + (schale_id * punkte_pro_schale)
        r = np.sqrt(n) * (0.2 * (schale_id + 1))
        winkel = n * w_gold
        
        maske_radius = (r >= innen_radius) & (r <= aussen_radius)
        n = n[maske_radius]
        r = r[maske_radius]
        winkel = winkel[maske_radius]
        
        anzahl_punkte = len(n)
        if anzahl_punkte < 3:
            continue
            
        b_wert = np.zeros(anzahl_punkte)
        maske_kern = n < 9
        b_wert[maske_kern] = basis_werte[n[maske_kern]]
        
        idx_aussen = (n[~maske_kern] - 1) % 8 + 1
        vorzeichen = np.where(n[~maske_kern] % 2 == 0, 1.0 / phi, -1.0 / phi)
        b_wert[~maske_kern] = basis_werte[idx_aussen] * vorzeichen
        
        x_basis = b_wert * phi * np.cos(winkel)
        y_basis = b_wert * phi * np.sin(winkel)
        
        # Wellen-Topologie
        winkel_deg = np.degrees(winkel) % 360
        segment = (winkel_deg / 45).astype(int) % 8
        z_basis = np.sin((r - innen_radius) / (aussen_radius - innen_radius) * np.pi)
        z_mod = np.where(segment % 2 == 0, z_basis * 1.5, z_basis * 2.2 + np.sin(winkel * 4 + t * 2.0) * 0.3)
        
        x_flach = x_basis - (1.0 / (r - innen_radius + 0.1) * np.cos(winkel) * 0.5)
        y_flach = y_basis - (1.0 / (r - innen_radius + 0.1) * np.sin(winkel) * 0.5)
        z_flach = z_mod + (np.cos(winkel - np.radians(45.0)) * 3.5)
        
        # Kugelprojektion
        r_flach = np.sqrt(x_flach**2 + y_flach**2) + 0.001
        theta = (r_flach / aussen_radius) * np.pi * 0.5
        phi_kugel = np.arctan2(y_flach, x_flach)
        
        radius_mit_struktur = kugel_radius + (z_flach * 0.2)
        
        x_kugel = radius_mit_struktur * np.sin(theta) * np.cos(phi_kugel)
        y_kugel = radius_mit_struktur * np.sin(theta) * np.sin(phi_kugel)
        z_kugel = radius_mit_struktur * np.cos(theta)
        
        # Gegenläufige Rotation
        dreh_richtung = 1.0 if schale_id % 2 == 0 else -1.0
        rot_winkel = np.radians(schale_id * 45.0) + (t * 2.0 * dreh_richtung)
        cos_r, sin_r = np.cos(rot_winkel), np.sin(rot_winkel)
        
        x_rot = x_kugel * cos_r - y_kugel * sin_r
        y_rot = x_kugel * sin_r + y_kugel * cos_r
        z_rot = z_kugel
        
        schalen_vertices = np.column_stack((x_rot, y_rot, z_rot))
        
        # Sonnenlicht-Berechnung
        normalen = schalen_vertices / np.linalg.norm(schalen_vertices, axis=1, keepdims=True)
        licht_intensitaet = np.sum(normalen * sonne_normiert, axis=1)
        licht_intensitaet = (licht_intensitaet + 1.0) / 2.0
        
        alle_vertices.append(schalen_vertices)
        alle_lichtwerte.append(licht_intensitaet)
        
        # Polygon-Flächen weben
        for i in range(anzahl_punkte - 2):
            alle_faces.append([vertex_offset + i, vertex_offset + i + 1, vertex_offset + i + 2])
        vertex_offset += anzahl_punkte
        
    v_gesamt = np.concatenate(alle_vertices)
    l_gesamt = np.concatenate(alle_lichtwerte)
    v_gesamt -= np.mean(v_gesamt, axis=0)
    
    return v_gesamt, alle_faces, l_gesamt

def exportiere_max_ausbau_obj(vertices, faces, dateiname="integrales_3d_modell.obj"):
    """Exportiert Vertices und Dreiecksflächen in eine fertige 3D-Geometriedatei."""
    with open(dateiname, "w") as f:
        f.write("# TGAY Maximaler Ausbau: Rotierende Kugel-Helix mit Polygonnetz\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        
        # Punkte schreiben
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        f.write("\n# Polygon-Flächen\n")
        # Flächen schreiben
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            
    print(f"===================================================================")
    print(f" 3D-Objektkörper generiert und als Festkörper-Geometrie gesichert.")
    print(f" Datei: {dateiname}")
    print(f"===================================================================")

# Ausführen des Systems und Generierung der Datei
vertices, faces, licht = generiere_4d_resonator_mit_sonnenlicht(t=1.0)
exportiere_max_ausbau_obj(vertices, faces)
