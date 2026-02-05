import cv2
import numpy as np
import math
import pyautogui

captura = cv2.VideoCapture(0)

degete_queue = []
degete_stabile = 0
contor_actiuni = 0

roi_x, roi_y = 150, 150
roi_latime, roi_inaltime = 250, 250

calibrat = False
piele_inferioara = None
piele_superioara = None

while True:
    ret, cadru = captura.read()
    if not ret: break
    img = cv2.resize(cadru, (600, 600))

    roi = img[roi_y: roi_y + roi_inaltime, roi_x: roi_x + roi_latime]
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_latime, roi_y + roi_inaltime), (255, 255, 255), 2)

    blur = cv2.GaussianBlur(roi, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if not calibrat:
        mostra_x, mostra_y = 75, 75
        mostra_latime, mostra_inaltime = 100, 100

        cv2.rectangle(roi, (mostra_x, mostra_y), (mostra_x + mostra_latime, mostra_y + mostra_inaltime), (0, 255, 0), 2)

        cv2.putText(img, "PUNE PALMA IN PATRAT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "APASA 'c' PENTRU CALIBRARE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            roi_mostra = hsv[mostra_y: mostra_y + mostra_inaltime, mostra_x: mostra_x + mostra_latime]

            medie_h = np.median(roi_mostra[:,:,0])
            medie_s = np.median(roi_mostra[:,:,1])
            medie_v = np.median(roi_mostra[:,:,2])

            marja_h = 15
            marja_s = 50
            marja_v = 50

            piele_inferioara = np.array([max(0, medie_h - marja_h), max(0, medie_s - marja_s), max(0, medie_v - marja_v)], dtype=np.uint8)
            piele_superioara = np.array([min(180, medie_h + marja_h), min(255, medie_s + marja_s), min(255, medie_v + marja_v)], dtype=np.uint8)

            print(f"Calibrat! Marginea inferioara: {piele_inferioara}, Marginea superioara: {piele_superioara}")
            calibrat = True

    else:
        masca = cv2.inRange(hsv, piele_inferioara, piele_superioara)
        masca = cv2.dilate(masca, None, iterations = 2)
        contururi, _ = cv2.findContours(masca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        degete_din_frame = 0

        if len(contururi) > 0:
            contur = max(contururi, key = cv2.contourArea)
            if cv2.contourArea(contur) > 2000:
                masca_palmei = np.zeros(masca.shape, dtype = np.uint8)
                cv2.drawContours(masca_palmei, [contur], -1, 255, -1)
                distanta = cv2.distanceTransform(masca_palmei, cv2.DIST_L2, 5)
                _, valoare_max, _, locatie_max = cv2.minMaxLoc(distanta)

                centru = locatie_max
                raza = int(valoare_max)
                cv2.circle(roi, centru, raza, (255, 255, 0), 2)
                limita_deget = raza * 2.0

                punctele_formei = cv2.convexHull(contur, returnPoints = True)
                coordonate_degete = []
                for pt in punctele_formei:
                    x, y = pt[0]
                    distanta_de_la_centru = math.sqrt((x - centru[0]) ** 2 + (y - centru[1]) ** 2)
                    if y < centru[1] and distanta_de_la_centru > limita_deget:
                        if all(math.sqrt((x - f[0]) ** 2 + (y - f[1]) ** 2) > 40 for f in coordonate_degete):
                            coordonate_degete.append((x, y))

                indicii_formei = cv2.convexHull(contur, returnPoints = False)
                defecte = cv2.convexityDefects(contur, indicii_formei)
                nr_defecte = 0
                if defecte is not None:
                    for i in range(defecte.shape[0]):
                        s, sf, f, d = defecte[i, 0]
                        start, sfarsit, departe = tuple(contur[s][0]), tuple(contur[sf][0]), tuple(contur[f][0])
                        a, b, c = math.dist(sfarsit, start), math.dist(departe, start), math.dist(sfarsit, departe)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                        if angle <= 90 and d / 256.0 > 20:
                            nr_defecte += 1
                            cv2.circle(roi, departe, 5, [0, 0, 255], -1)

                nr_total_degete = len(coordonate_degete)
                if nr_defecte: nr_total_degete = nr_defecte + 1
                if len(coordonate_degete) == 1 and nr_defecte == 0:
                    for pt in coordonate_degete:
                        cv2.circle(roi, pt, 8, (0, 255, 0), -1)

                degete_din_frame = nr_total_degete

        degete_queue.append(degete_din_frame)
        if len(degete_queue) > 15: degete_queue.pop(0)
        if len(degete_queue) > 0:
            numar = {val: degete_queue.count(val) for val in set(degete_queue)}
            degete_stabile = max(numar, key=numar.get)

        contor_actiuni += 1

        if contor_actiuni > 20:
            if degete_stabile == 1:
                pyautogui.press('volumedown')
                print("Action: Volume Down")
                contor_actiuni = 15
            elif degete_stabile == 3:
                pyautogui.press('volumeup')
                print("Action: Volume Up")
                contor_actiuni = 15
            elif degete_stabile == 5:
                pyautogui.press('playpause')
                print("Action: Play/Pause")
                contor_actiuni = 0
                contor_actiuni = -20
            elif degete_stabile == 4:
                pyautogui.hotkey('delete')
                print("Action: Discord Mute/Unmute")
                contor_actiuni = -20

        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            calibrat = False

    if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.imshow('img', img)

captura.release()
cv2.destroyAllWindows()



