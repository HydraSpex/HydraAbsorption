#HydraAbsorption 1.0

#Info ---------------------------------------------------------------
VERSION = "1.0 Testversion"
print("Version: HydraAbsorption " + str(VERSION))
UPDATES = "APDs new performance\n\tLive Plot Window\n\tLive Plot in Measurement\n\tminor fixes"
NUMUPDATES = 4
print("Updates: " + UPDATES)
COPYRIGHT = "Property of University Tübingen"
print("Copyright: " + COPYRIGHT)
CONTACT = "info@hydraspex.com"
print("Contact: " + CONTACT)
CITE = "Available soon!"
print("If HydraAbsorption contributes to publish a work please cite: " + CITE + "\n\n")


#Imports ---------------------------------------------------------------
import os
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, savgol_filter
import math
#from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
#                             QFileDialog, QMessageBox, QHBoxLayout, QSizePolicy,
#                             QGridLayout, QSpinBox, QGroupBox, QCheckBox)
#from PyQt5.QtCore import QThread, pyqtSignal, QWaitCondition, QMutex
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
#import matplotlib as mpl
#mpl.rcParams.update({'text.color': "white",
#                        'axes.labelcolor': "white",
#                        'xtick.color': "white",
#                        'ytick.color': "white"})
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
plt.rcParams.update({'font.size': 10})
plt.rcParams['svg.fonttype'] = 'none'


#from numba import jit

#Global Variables ---------------------------------------------------------------
StyleName = "Fusion"            #'Fusion', 'Windows', 'windowsvista'(['bb10dark', 'bb10bright', 'cleanlooks', 'cde', 'motif', 'plastique', 'Windows', 'Fusion'])
#StyleName = "Windows"
#StyleName = "windowsvista"
StyleColor = "dark"             #'light', 'dark'
#StyleColor = "light"

WindowPosX = 15
WindowPosY = 50
WindowWidth = 1000
WindowHeight = 900
CurrentPage = 0

Pixel = 128
Spectra = 80
AbsMin = 0
AbsMax = 1.6
Fixed2d = True
Fixed3d = True
Fixed3dInit = False
Cb3d = True
InitMin = 0
InitMax = 1000
SaveSVG = False
PixelPics = False
SelectPos = 0


#Worker Thread ---------------------------------------------------------------
class WorkerThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    progress2 = pyqtSignal(int)
    progress3 = pyqtSignal(int)
    plot_bereit = pyqtSignal(
        np.ndarray)  # Signal für den ersten Plot, jetzt mit np.ndarray

    def __init__(self, ordnerpfad, RangePos, OffsetX, OffsetY, x_center, y_center, x_mittel_center,
                 y_mittel_center, size_mittel, lowerLimit, upperLimit, bereich_mittelwert_range, klick_position_event, mutex):
        super().__init__()
        global Spectra
        self.ordnerpfad = ordnerpfad
        self.RangePos = RangePos
        self.OffsetX = OffsetX
        self.OffsetY = OffsetY
        self.x_center = x_center
        self.y_center = y_center
        self.x_mittel_center = x_mittel_center
        self.y_mittel_center = y_mittel_center
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.size_mittel = size_mittel
        self.bereich_mittelwert_range = bereich_mittelwert_range
        self.klick_position_event = klick_position_event
        self.mutex = mutex

    def run(self):
        self.plot_ordner(self.ordnerpfad)
        self.plot_erste_werte(self.ordnerpfad)
        self.plot_max(self.ordnerpfad)
        self.finished.emit()

    def lade_daten(self, dateiname):
        global Pixel
        
        try:
            with open(dateiname, 'r') as f:
                # Die ersten drei Zeilen überspringen
                for _ in range(3):
                    next(f)

                # Die nächsten 128 Zeilen einlesen
                daten = []
                for _ in range(Pixel):
                    zeile = f.readline()
                    if zeile:
                        daten.append(list(map(float, zeile.strip().split('\t'))))
                    else:
                        break  # Dateiende erreicht

            return np.array(daten)
        except FileNotFoundError:
            print(f"Fehler: Datei '{dateiname}' nicht gefunden.")
            return None

    def plot_max(self, ordnerpfad):
        ordnerpfad = ordnerpfad + "/"
        newordnerpfad = ordnerpfad + "Spectra/"
        FilepathLocalMax = newordnerpfad + "LocalMax"
        FilepathLocalMin = newordnerpfad + "LocalMin"
        FilepathGlobalMax = newordnerpfad + "GlobalMax.png"
        FilepathGlobalMin = newordnerpfad + "GlobalMin.png"
        #print(ordnerpfad)
        #print(newordnerpfad)


        #print("All Maxima:")
        #print(self.max_all)
        #print(len(self.max_all))
        Shape = int(math.sqrt(len(self.max_all)))
        
        #print("All Maxima New:")
        #print(self.max_all_new)
        #print(len(self.max_all_new))
        i = 0
        LenMax = 0
        LenMaxLowest = 100
        while i < len(self.max_all_new):
            if len(self.max_all_new[i]) > LenMax:
                LenMax = len(self.max_all_new[i])
            if len(self.max_all_new[i]) < LenMaxLowest:
                LenMaxLowest = len(self.max_all_new[i])
            i += 1
        #print("LenMax: " + str(LenMax))
        #print("LenMaxLowest: " + str(LenMaxLowest))
        #print("All Minima New:")
        #print(self.min_all_new)
        #print(len(self.min_all_new))
        i = 0
        LenMin = 0
        while i < len(self.min_all_new):
            if len(self.min_all_new[i]) > LenMin:
                LenMin = len(self.min_all_new[i])
            i += 1
        #print(LenMin)

        ShapeNew = int(math.sqrt(len(self.max_all_new)))
        #print("ShapeNew: " + str(ShapeNew))
        #print(self.max_all_new)


        max_all_new_single = []
        i = 0
        while i < LenMax:
            j = 0
            Lines = []
            while j < len(self.max_all_new):
                if i < len(self.max_all_new[j]):
                    Lines.append(self.max_all_new[j][i])
                else:
                    Lines.append(self.max_all_new[j][len(self.max_all_new[j])-1])
                #print(i,j)
                #print(Lines)
                j += 1
            max_all_new_single.append(Lines)
            fortschritt = int((i + 1) / LenMax * 49)
            self.progress3.emit(fortschritt)
            i += 1

        i = 0
        while i < LenMax:
            DataMaxNew = np.reshape(max_all_new_single[i], (ShapeNew, ShapeNew))
            #print(self.max_all_new)
            anzahl_zeilen, anzahl_spalten = DataMaxNew.shape
            x = np.arange(anzahl_spalten)
            y = np.arange(anzahl_zeilen)
            X, Y = np.meshgrid(x, y)

            fig_max_all = Figure()
            #canvas_max_all = FigureCanvas(fig_max_all)
            fig_max_all, ax_max_all  = plt.subplots()
            mesh_max_all = ax_max_all.pcolormesh(X, Y, DataMaxNew, cmap='Greys')
            fig_max_all.colorbar(mesh_max_all)
            ax_max_all.set_xlabel('X')
            ax_max_all.set_ylabel('Y')
            ax_max_all.set_title("Local maximum " + str(i))
            fig_max_all.savefig(FilepathLocalMax + "_" + str(i) + ".png")
            fortschritt = 49 + int((i + 1) / LenMax * 50)
            self.progress3.emit(fortschritt)
            i += 1

        DataMax = np.reshape(self.max_all, (Shape, Shape))
        anzahl_zeilen, anzahl_spalten = DataMax.shape
        x = np.arange(anzahl_spalten)
        y = np.arange(anzahl_zeilen)
        X, Y = np.meshgrid(x, y)

        fig_max_all = Figure()
        #canvas_max_all = FigureCanvas(fig_max_all)
        fig_max_all, ax_max_all  = plt.subplots()
        mesh_max_all = ax_max_all.pcolormesh(X, Y, DataMax, cmap='Greys')
        #mesh_max_all = ax_max_all.pcolormesh(X, Y, DataMax, cmap='Greys')
        fig_max_all.colorbar(mesh_max_all)
        ax_max_all.set_xlabel('X')
        ax_max_all.set_ylabel('Y')
        ax_max_all.set_title(
            f'Max all')  # Dateiname im Titel verwenden
        fig_max_all.savefig(FilepathGlobalMax)
        self.progress3.emit(100)


    def plot_daten(self, dateipfad, daten_array):
        if daten_array is None:
            return

        # Datenarray emittieren
        self.plot_bereit.emit(daten_array)

    def plot_daten_mit_bereich(self, dateipfad, daten_array):
        global Pixel
        global InitMin
        global InitMax
        global Spectra
        global AbsMin
        global AbsMax
        global Fixed2d
        global Fixed3d
        global Fixed3dInit
        global Cb3d
        global SaveSVG

        if daten_array is None:
            return None

        anzahl_zeilen, anzahl_spalten = daten_array.shape
        x = np.arange(anzahl_spalten)
        y = np.arange(anzahl_zeilen)
        X, Y = np.meshgrid(x, y)


        # Bereich für die Mittelwertbildung definieren
        x_mittel_start = self.x_mittel_center - self.size_mittel // 2
        x_mittel_end = self.x_mittel_center + self.size_mittel // 2
        y_mittel_start = self.y_mittel_center - self.size_mittel // 2
        y_mittel_end = self.y_mittel_center + self.size_mittel // 2
        bereich_mittelwert_zeilen = slice(y_mittel_start, y_mittel_end)
        bereich_mittelwert_spalten = slice(x_mittel_start, x_mittel_end)
        #print("bereich_mittelwert_zeilen: " + str(bereich_mittelwert_zeilen))
        

        # Daten im Mittelwertbereich extrahieren
        bereich_mittelwert_daten = daten_array[
            bereich_mittelwert_zeilen, bereich_mittelwert_spalten]

        # Mittelwert berechnen
        mittelwert = np.mean(bereich_mittelwert_daten)

        # Bereich für den zweiten Plot basierend auf der Klickposition definieren
        self.x_center = self.x_center + self.OffsetX
        self.y_center = self.y_center + self.OffsetY
        if self.RangePos == 0:
            SlicerZeileY = self.y_center - int(self.bereich_mittelwert_range/2)
            SlicerZeileX = self.y_center + int(self.bereich_mittelwert_range/2)
            SlicerSpalteY = self.x_center - int(self.bereich_mittelwert_range/2)
            SlicerSpalteX = self.x_center + int(self.bereich_mittelwert_range/2)
        elif self.RangePos == 1:
            SlicerZeileY = self.y_center
            SlicerZeileX = self.y_center+self.bereich_mittelwert_range
            SlicerSpalteY = self.x_center-self.bereich_mittelwert_range
            SlicerSpalteX = self.x_center
            #bereich_plot_zeilen = slice(self.y_center, self.y_center+self.bereich_mittelwert_range)
            #bereich_plot_spalten = slice(self.x_center-self.bereich_mittelwert_range, self.x_center)
        elif self.RangePos == 2:
            SlicerZeileY = self.y_center-self.bereich_mittelwert_range
            SlicerZeileX = self.y_center
            SlicerSpalteY = self.x_center-self.bereich_mittelwert_range
            SlicerSpalteX = self.x_center
            #bereich_plot_zeilen = slice(self.y_center-self.bereich_mittelwert_range, self.y_center)
            #bereich_plot_spalten = slice(self.x_center-self.bereich_mittelwert_range, self.x_center)
        elif self.RangePos == 3:
            SlicerZeileY = self.y_center
            SlicerZeileX = self.y_center+self.bereich_mittelwert_range
            SlicerSpalteY = self.x_center
            SlicerSpalteX = self.x_center+self.bereich_mittelwert_range
            #bereich_plot_zeilen = slice(self.y_center, self.y_center+self.bereich_mittelwert_range)
            #bereich_plot_spalten = slice(self.x_center, self.x_center+self.bereich_mittelwert_range)
        elif self.RangePos == 4:
            SlicerZeileY = self.y_center
            SlicerZeileX = self.y_center+self.bereich_mittelwert_range
            SlicerSpalteY = self.x_center-self.bereich_mittelwert_range
            SlicerSpalteX = self.x_center
            #bereich_plot_zeilen = slice(self.y_center, self.y_center+self.bereich_mittelwert_range)
            #bereich_plot_spalten = slice(self.x_center-self.bereich_mittelwert_range, self.x_center)

        if SlicerZeileY < 0:
            SlicerZeileX = SlicerZeileX - SlicerZeileY
            SlicerZeileY = 0
        if SlicerZeileX > (Pixel-1):
            SlicerZeileY = SlicerZeileY + ((Pixel-1) - SlicerZeileX)
            SlicerZeileX = (Pixel-1)
        if SlicerSpalteY < 0:
            SlicerSpalteX = SlicerSpalteX - SlicerSpalteY
            SlicerSpalteY = 0
        if SlicerSpalteX > (Pixel-1):
            SlicerSpalteY = SlicerSpalteY + ((Pixel-1) - SlicerSpalteX)
            SlicerSpalteX = (Pixel-1)
        bereich_plot_zeilen = slice(SlicerZeileY, SlicerZeileX)
        bereich_plot_spalten = slice(SlicerSpalteY, SlicerSpalteX)
        #print(bereich_plot_zeilen, bereich_plot_spalten)

        # Daten im Plotbereich extrahieren
        bereich_plot_daten = daten_array[bereich_plot_zeilen,bereich_plot_spalten]

        # Erster Plot (vollständiger Meshgrid mit markierten Bereichen)
        fig1 = Figure()
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111)
        if Fixed3dInit:
            mesh1 = ax1.pcolormesh(X, Y, daten_array, cmap='Greys', vmin=InitMin, vmax=InitMax)
        else:
            mesh1 = ax1.pcolormesh(X, Y, daten_array, cmap='Greys')
        if Cb3d:
            fig1.colorbar(mesh1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(
            f'Full range at {os.path.basename(dateipfad)}')  # Dateiname im Titel verwenden

        # Roten Kasten um den Mittelwertbereich zeichnen
        rect_mittelwert = plt.Rectangle(
            (bereich_mittelwert_spalten.start, bereich_mittelwert_zeilen.start),
            bereich_mittelwert_spalten.stop - bereich_mittelwert_spalten.start,
            bereich_mittelwert_zeilen.stop - bereich_mittelwert_zeilen.start,
            fill=False,
            edgecolor='yellow',
            linewidth=2)
        ax1.add_patch(rect_mittelwert)

        # Blauen Kasten um den Plotbereich zeichnen
        rect_plot = plt.Rectangle(
            (bereich_plot_spalten.start, bereich_plot_zeilen.start),
            bereich_plot_spalten.stop - bereich_plot_spalten.start,
            bereich_plot_zeilen.stop - bereich_plot_zeilen.start,
            fill=False,
            edgecolor='blue',
            linewidth=2)
        ax1.add_patch(rect_plot)

        # Plot im selben Ordner wie die Quelldaten speichern
        ordnerpfad = os.path.dirname(dateipfad)
        dateiname = os.path.basename(dateipfad)
        if SaveSVG:
            plot_dateiname1 = os.path.join(ordnerpfad, os.path.splitext("Full_" + dateiname)[0] + ".svg")
        else:
            plot_dateiname1 = os.path.join(ordnerpfad, os.path.splitext("Full_" + dateiname)[0] + ".png")
        fig1.savefig(plot_dateiname1)

        # Zweiter Plot (Ausschnitt des Plotbereichs in 64x64 Pixeln)
        fig2 = Figure()
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        img = ax2.imshow(bereich_plot_daten,
                         cmap='Greys',
                         interpolation='nearest')
        # Renderer von fig2 verwenden
        bereich_daten_interpoliert = np.array(
            img.make_image(canvas2.get_renderer())[0][:, :, 0])
        #mesh2 = ax2.pcolormesh(bereich_daten_interpoliert, cmap='viridis')
        if Fixed3dInit:
            mesh2 = ax2.pcolormesh(bereich_plot_daten, cmap='Greys', vmin=InitMin, vmax=InitMax)
        else:
            mesh2 = ax2.pcolormesh(bereich_plot_daten, cmap='Greys')
        if Cb3d:
            fig2.colorbar(mesh2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(
            f'Area at {os.path.basename(dateipfad)}')  # Dateiname im Titel verwenden

        # Plot im selben Ordner wie die Quelldaten speichern
        if SaveSVG:
            plot_dateiname2 = os.path.join(ordnerpfad, os.path.splitext("Area_" + dateiname)[0] + ".svg")
        else:
            plot_dateiname2 = os.path.join(ordnerpfad, os.path.splitext("Area_" + dateiname)[0] + ".png")
        fig2.savefig(plot_dateiname2)



        # Dritter Plot (Ausschnitt des Plotbereichs in 64x64 Pixeln)
        #AbsorbanceArray = bereich_daten_interpoliert
        AbsorbanceArray = np.log(mittelwert/(bereich_plot_daten+1))
        #print(mittelwert)

        fig3 = Figure()
        canvas3 = FigureCanvas(fig3)
        ax3 = fig3.add_subplot(111)
        img = ax3.imshow(AbsorbanceArray,
                         cmap='Greys',
                         interpolation='nearest')
        # Renderer von fig2 verwenden
        AbsorbanceArray_interpoliert = np.array(
            img.make_image(canvas3.get_renderer())[0][:, :, 0])
        #mesh3 = ax3.pcolormesh(AbsorbanceArray_interpoliert, cmap='viridis')
        if Fixed3d:
            mesh3 = ax3.pcolormesh(AbsorbanceArray, cmap='Greys',vmin=AbsMin, vmax=AbsMax)
        else:
            mesh3 = ax3.pcolormesh(AbsorbanceArray, cmap='Greys')
        if Cb3d:
            cbar =  fig3.colorbar(mesh3)
        #cbar = plt.colorbar(heatmap)
            cbar.set_label('BG corr. Absorbance (OD)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title(
            f'Absorbance at {os.path.basename(dateipfad)}')  # Dateiname im Titel verwenden
        #ax3.set_zlim(1, 3)

        # Plot im selben Ordner wie die Quelldaten speichern
        if SaveSVG:
            plot_dateiname3 = os.path.join(ordnerpfad, os.path.splitext("Absorbance_" + dateiname)[0] + ".svg")
        else:
            plot_dateiname3 = os.path.join(ordnerpfad, os.path.splitext("Absorbance_" + dateiname)[0] + ".png")
        fig3.savefig(plot_dateiname3)

        return mittelwert, AbsorbanceArray, bereich_plot_daten

    def plot_ordner(self, ordnerpfad):
        global Pixel

        newordnerpfad = ordnerpfad + "/Spectra/"
        try:
            os.makedirs(newordnerpfad)
            print("Data folder created")
        except:
            pass

        mittelwerte = []
        dateien = [
            f for f in os.listdir(ordnerpfad) if f.endswith(".dat")
        ]
        anzahl_dateien = len(dateien)
        for i, dateiname in enumerate(dateien):
            dateipfad = os.path.join(ordnerpfad, dateiname)
            daten_array = self.lade_daten(dateipfad)
            
            Pixel = len(daten_array)
            #print("Pixels: " + str(Pixel))
            # Plot zur Bereichsauswahl erstellen
            self.plot_daten(dateipfad, daten_array)
            # Warten, bis der Benutzer den Bereich im Hauptthread ausgewählt hat
            self.mutex.lock()
            self.klick_position_event.wait(self.mutex)
            self.mutex.unlock()
            # Plots mit ausgewähltem Bereich erstellen
            mittelwert, AbsorbanceArray, Daten_Bereich = self.plot_daten_mit_bereich(dateipfad, daten_array)
            if mittelwert is not None:
                mittelwerte.append(mittelwert)

            wavelength = dateiname[:len(dateiname)-4]
            filename = ordnerpfad + "/" + str(wavelength) + ".txt"
            self.Txt_Messfile = open(filename, "w")
            filename2 = newordnerpfad + str(wavelength) + ".overview"
            self.Txt_Messfile2 = open(filename2, "w")
            j = 0
            while j < len(AbsorbanceArray):
                k = 0
                while k < len(AbsorbanceArray):
                    self.Txt_Messfile.write(str(AbsorbanceArray[j][k]))
                    self.Txt_Messfile2.write(str(int(Daten_Bereich[j][k])))
                    if k < len(AbsorbanceArray)-1:
                        self.Txt_Messfile.write("\t")
                        self.Txt_Messfile2.write("\t")
                    k += 1
                if j < len(AbsorbanceArray)-1:
                    self.Txt_Messfile.write("\n")
                    self.Txt_Messfile2.write("\n")
                j += 1
            self.Txt_Messfile.close()
            self.Txt_Messfile2.close()

            fortschritt = int((i + 1) / anzahl_dateien * 100)
            self.progress.emit(fortschritt)
        return mittelwerte
    
    #@jit
    def plot_erste_werte(self, ordnerpfad):
        global Spectra
        global AbsMin
        global AbsMax
        global Fixed2d
        global SaveSVG
        global PixelPics
            
        #print("Spectra: " + str(Spectra))
       
        ordnerpfad = ordnerpfad + "/"
        newordnerpfad = ordnerpfad + "Spectra/"
        #print(ordnerpfad)
        #print(newordnerpfad)
        try:
            os.makedirs(newordnerpfad)
            print("Data folder created")
        except:
            pass
        i = 0
        self.max_all = []
        self.min_all_new = []
        self.max_all_new = []
        while i < Spectra:
            j = 0
            while j < Spectra:
                erste_werte = []
                dateinamen = []
                WLs = []
                
                filename = newordnerpfad + "Pixel_" + str(i) + "x" + str(j) + ".txt"
                self.Txt_Pixel = open(filename, "w")

                for dateiname in os.listdir(ordnerpfad):
                    #print(ordnerpfad)
                    if dateiname.endswith(".txt"):
                        dateipfad = os.path.join(ordnerpfad, dateiname)
                        #print(dateipfad)
                        try:
                            with open(dateipfad, 'r') as f:
                                alle_zeilen = f.readlines()
                                erste_zeile = alle_zeilen[i]
                                #print(type(erste_zeile))
                                #print(erste_zeile)
                                #erster_wert = erste_zeile.split('\t')[31]  # Extrahiere den ersten Wert
                                #print(erster_wert)
                                erster_wert = float(erste_zeile.split('\t')[j])  # Extrahiere den ersten Wert
                                erste_werte.append(erster_wert)
                                dateinamen.append(dateiname)
                                WL = dateiname[:len(dateiname)-4]
                                WLs.append(WL)
                                self.Txt_Pixel.write(str(WL) + "\t" + str(erster_wert) + "\n")
                        except FileNotFoundError:
                            print(f"Error: File '{dateipfad}' not found.")
                        except IndexError:
                            print(f"Error: File '{dateipfad}' has no tabs.")
                        except ValueError:
                            print(f"Error: First value '{dateipfad}' is not a number.")

                #print(i, j)
                self.Txt_Pixel.close()
                #print(type(erste_werte))
                #print(erste_werte)

                npa = np.asarray(erste_werte, dtype=np.float32)
                npamax = argrelextrema(npa, np.greater)
                npamin = argrelextrema(npa, np.less)
                #print(npamax)
                #print(npamax[0])
                #print(npamin)

                if erste_werte:
                    self.max_all.append(int(WLs[erste_werte.index(max(erste_werte))]))
                    k = 0
                    l = 0
                    maxlocal = []
                    minlocal = []
                    while k < len(npamax[0]):
                        maxlocal.append(int(WLs[npamax[0][k]]))
                        k += 1 
                    self.max_all_new.append(maxlocal)
                    while l < len(npamin[0]):
                        minlocal.append(int(WLs[npamin[0][l]]))
                        l += 1 
                    self.min_all_new.append(minlocal)

                    if PixelPics:
                        # 2D-Plot erstellen        
                        plt.plot(erste_werte)
                        if Fixed2d:
                            plt.ylim((AbsMin, AbsMax))
                        plt.grid()
                        plt.xlabel("Wavelength [nm]")
                        plt.ylabel("BG corr. Absorbance [OD]")
                        plt.title("Absorbance Profile of Pixel " + str(i) + "x" + str(j))
                        plt.xticks(range(len(dateinamen)), WLs, rotation=90, ha='right')  # Dateinamen als x-Ticks
                        plt.tight_layout()  # Optimiert die Layout-Abstände
                        if SaveSVG:
                            plt.savefig(newordnerpfad + "Spectrum_" + str(i) + "_" + str(j) + ".svg")
                        else:
                            plt.savefig(newordnerpfad + "Spectrum_" + str(i) + "_" + str(j) + ".png")
                        plt.clf()
                        #plt.show()
                else:
                    print("No valid txt-File found.")
                
                erste_werte.clear()
                dateinamen.clear()
                j += 1
            i += 1
            
            if erste_werte:
                if PixelPics:
                    # 2D-Plot erstellen        
                    plt.plot(erste_werte)
                    if Fixed2d:
                        plt.ylim((AbsMin, AbsMax))
                    plt.grid()
                    plt.xlabel("Wavelength [nm]")
                    plt.ylabel("BG corr. Absorbance [OD]")
                    plt.title("Absorbance Profile of all Pixels")
                    plt.xticks(range(len(dateinamen)), WLs, rotation=90, ha='right')  # Dateinamen als x-Ticks
                    plt.tight_layout()  # Optimiert die Layout-Abstände
                    plt.savefig(newordnerpfad + "Spectrum_all.png")
                    if SaveSVG:
                        plt.savefig(newordnerpfad + "Spectrum_all.svg")
                    else:
                        plt.savefig(newordnerpfad + "Spectrum_all.png")
                    plt.clf()
                    #plt.show()
            else:
                print("No valid txt-File found.")
            
            fortschritt = int((i + 1) / Spectra * 100)
            self.progress2.emit(fortschritt)

            #return max_all

#Main Window ---------------------------------------------------------------
class MyWindow(QWidget):
    klick_position_event = QWaitCondition()
    mutex = QMutex()  # Mutex erstellen

    def __init__(self):
        super().__init__()
        self.clicked_points = [] # Liste zum Speichern der Klick-Koordinaten
        self.z_data = None       # Zum Speichern der geladenen Daten
        self.clicked_points3 = [] # Liste zum Speichern der Klick-Koordinaten
        self.z_data3 = None       # Zum Speichern der geladenen Daten
        self.X = None            # Zum Speichern der X-Koordinaten des Gitters
        self.Y = None            # Zum Speichern der Y-Koordinaten des Gitters
        self.mesh_plot = None    # Referenz auf das pcolormesh-Objekt
        self.mesh_plot3 = None    # Referenz auf das pcolormesh-Objekt
        self.initUI()
        self.klick_position = None
        self.klick_position3 = None

    def initUI(self):
        global Pixel

        self.setWindowTitle('HydraAbsorption')
        self.setGeometry(15, 50, 1100, 950)
        self.setMinimumSize(QSize(1100,950))   

        #Tabs definieren
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        #für weiter Tabs hier eine neue Zeile einfügen

        #Tabs zum Widget hinzufügen
        self.tabs.addTab(self.tab1, "Absorption Mapping")
        self.tabs.addTab(self.tab2, "Spectra from Pixel")
        self.tabs.addTab(self.tab3, "Spectral Mapping")
        self.tabs.setCurrentIndex(0) 

        #Darkmode ist essenziell
        if StyleColor == "dark" and StyleName == "windowsvista":
                self.tabs.setStyleSheet("color: black;"
                                        "background-color: rgb(153,153,153);")
                
        self.layout = QVBoxLayout(self)

        
        #------------------------ Tab1 -----------------------------
        self.tab1.grid = QGridLayout()
        #self.setLayout(grid)                                                                                                #Icon oben links

        # Plots generieren Button
        self.plot_button = QPushButton("Generate plots", self)
        self.plot_button.clicked.connect(self.plots_generieren)
        self.plot_button.setToolTip("Start plotting")                                                   #Setzt eine Buttonbeschreibung bei MouseOver
        self.plot_button.setStyleSheet("color: black; background-color: rgb(0,255,0); border-radius: 10px")
        self.plot_button.setFixedHeight(50)
        self.plot_button.setFixedWidth(320)
        self.tab1.grid.addWidget(self.plot_button, 0, 0)

        # Plotrange definieren
        self.PlotRange = QComboBox(self)
        #self.PlotRange.setFixedSize(210, 30)
        self.PlotRange.addItem("Center")
        self.PlotRange.addItem("Upper left")
        self.PlotRange.addItem("Upper right")
        self.PlotRange.addItem("Lower left")
        self.PlotRange.addItem("Lower right")
        self.PlotRange.setCurrentIndex(0)
        self.PlotRange.setToolTip("Set the Startposition for the Plotrange.")
        #self.PlotRange.currentIndexChanged.connect(self.PlotRangeChange) 
        #self.PlotRange.setFont(QFont(self.Fontstyle, self.Fontsize, QFont.Bold))

        # Datarange Groupbox
        limits_groupbox = QGroupBox("Limits", self)
        self.tab1.grid.addWidget(limits_groupbox, 2, 0)
        #limits_layout = QVBoxLayout(limits_groupbox)
        limits_layout_grid = QGridLayout(limits_groupbox)

        self.upper_spinbox = QDoubleSpinBox(self)
        self.upper_spinbox.setRange(-100, 100)  # Bereich angepasst
        self.upper_spinbox.setValue(2)
        self.upper_spinbox.setDecimals(2)
        self.upper_spinbox.setSingleStep(0.01)
        self.upper_spinbox.valueChanged.connect(self.BereichChanges)
        self.upper_spinbox.setToolTip("Set the upper Limit for the Absorption Plots.")
        self.lower_spinbox = QDoubleSpinBox(self)
        self.lower_spinbox.setRange(-100, 100)  # Bereich angepasst
        self.lower_spinbox.setValue(0)
        self.lower_spinbox.setDecimals(2)
        self.lower_spinbox.setSingleStep(0.01)
        self.lower_spinbox.valueChanged.connect(self.BereichChanges)
        self.lower_spinbox.setToolTip("Set the lower Limit for the Absorption Plots.")

        limits_layout_grid.addWidget(QLabel("Upper Abs. limit [OD]:", self), 0, 1)
        limits_layout_grid.addWidget(QLabel("Lower Abs. limit [OD]:", self), 0, 0)

        limits_layout_grid.addWidget(self.upper_spinbox, 1, 1)
        limits_layout_grid.addWidget(self.lower_spinbox, 1, 0)

        self.upper_spinbox_Init = QSpinBox(self)
        self.upper_spinbox_Init.setRange(0, 1000000)  # Bereich angepasst
        self.upper_spinbox_Init.setValue(1000)
        self.upper_spinbox_Init.valueChanged.connect(self.BereichChanges)
        self.upper_spinbox_Init.setToolTip("Set the upper Limit for the normal Plots.")
        self.lower_spinbox_Init = QSpinBox(self)
        self.lower_spinbox_Init.setRange(0, 1000000)  # Bereich angepasst
        self.lower_spinbox_Init.setValue(0)
        self.lower_spinbox_Init.valueChanged.connect(self.BereichChanges)
        self.lower_spinbox_Init.setToolTip("Set the lower Limit for the normal Plots.")

        limits_layout_grid.addWidget(QLabel("Upper full limit [Counts]:", self), 2, 1)
        limits_layout_grid.addWidget(QLabel("Lower full limit [Counts]:", self), 2, 0)
        limits_layout_grid.addWidget(self.upper_spinbox_Init, 3, 1)
        limits_layout_grid.addWidget(self.lower_spinbox_Init, 3, 0)


        self.limit3dInit_check = QCheckBox("Fixed full limits 3D", self)
        self.limit3dInit_check.setChecked(True)
        self.limit3dInit_check.stateChanged.connect(self.BereichChanges)
        self.limit3dInit_check.setToolTip("If checked: All normal plots will have fixed upper and lower limits.")
        self.limits3d_check = QCheckBox("Fixed Abs. limits 3D", self)
        self.limits3d_check.setChecked(True)
        self.limits3d_check.stateChanged.connect(self.BereichChanges)
        self.limits3d_check.setToolTip("If checked: All 3D absorption plots will have fixed upper and lower limits.")
        self.limits2d_check = QCheckBox("Fixed Abs. limits 2D", self)
        self.limits2d_check.setChecked(True)
        self.limits2d_check.stateChanged.connect(self.BereichChanges)
        self.limits2d_check.setToolTip("If checked: All 2D absorption plots will have fixed upper and lower limits.")
        self.colorbar_check = QCheckBox("Colorbar", self)
        self.colorbar_check.setChecked(True)
        self.colorbar_check.stateChanged.connect(self.BereichChanges)
        self.colorbar_check.setToolTip("If checked: All 3D plots will have a colorbar.")
        self.SVG_check = QCheckBox("Save as SVG", self)
        self.SVG_check.setChecked(False)
        self.SVG_check.stateChanged.connect(self.BereichChanges)
        self.SVG_check.setToolTip("If checked: all plots will be saved as svg-Files.")
        self.PixelPics_check = QCheckBox("Save all Pixel", self)
        self.PixelPics_check.setChecked(False)
        self.PixelPics_check.stateChanged.connect(self.BereichChanges)
        self.PixelPics_check.setToolTip("If checked: All pixel data will be plotted. This will take a wile.")

        limits_layout_grid.addWidget(self.limits2d_check, 4, 0)
        limits_layout_grid.addWidget(self.limits3d_check, 4, 1)
        limits_layout_grid.addWidget(self.limit3dInit_check, 5, 0)
        limits_layout_grid.addWidget(self.PixelPics_check, 5, 1)
        limits_layout_grid.addWidget(self.colorbar_check, 6, 0)
        limits_layout_grid.addWidget(self.SVG_check, 6, 1)
        
        #limits_layout.addLayout(limits_layout_grid)
        #self.colorbar_check.stateChanged.connect(self.SomeChanges)
        #limits_layout.addWidget(self.limits2d_check)
        #limits_layout.addWidget(self.limits3d_check)
        #limits_layout.addWidget(self.limit3dInit_check)
        #limits_layout.addWidget(self.colorbar_check)
        #limits_layout.addWidget(self.SVG_check)

        self.x_Offset = QSpinBox(self)
        self.x_Offset.setRange(-1024, 1024)  # Bereich angepasst
        self.x_Offset.setValue(0)
        self.x_Offset.valueChanged.connect(self.BereichChanges)
        self.x_Offset.setToolTip("Sets an Offset to the your clicked plotrange position.")
        
        self.y_Offset = QSpinBox(self)
        self.y_Offset.setRange(-1024, 1024)  # Bereich angepasst
        self.y_Offset.setValue(0)
        self.y_Offset.valueChanged.connect(self.BereichChanges)
        self.y_Offset.setToolTip("Sets an Offset to the your clicked plotrange position.")

        # Plotbereich Groupbox
        plotbereich_groupbox = QGroupBox("Plotrange", self)
        self.tab1.grid.addWidget(plotbereich_groupbox, 3, 0)
        plotbereich_layout_grid = QGridLayout(plotbereich_groupbox)
        plotbereich_layout_grid.addWidget(self.PlotRange, 0, 0)
        plotbereich_layout_grid.addWidget(QLabel("X-Offset:", self), 1, 0)
        plotbereich_layout_grid.addWidget(self.x_Offset, 2, 0)
        plotbereich_layout_grid.addWidget(QLabel("Y-Offset:", self), 1, 1)
        plotbereich_layout_grid.addWidget(self.y_Offset, 2, 1)
        self.x_spinbox = QSpinBox(self)
        self.x_spinbox.setRange(0, (Pixel-1))  # Bereich angepasst
        self.x_spinbox.setValue(int(Pixel/2))
        self.x_spinbox.valueChanged.connect(self.BereichChanges)
        self.x_spinbox.setToolTip("Sets the plotrange. It is calculated automatically!")
        plotbereich_layout_grid.addWidget(QLabel("X-Position:", self), 3, 0)
        plotbereich_layout_grid.addWidget(self.x_spinbox, 4, 0)
        self.y_spinbox = QSpinBox(self)
        self.y_spinbox.setRange(0, (Pixel-1))  # Bereich angepasst
        self.y_spinbox.setValue(int(Pixel/2))
        self.y_spinbox.valueChanged.connect(self.BereichChanges)
        self.y_spinbox.setToolTip("Sets the plotrange. It is calculated automatically!")
        plotbereich_layout_grid.addWidget(QLabel("Y-Position:", self), 3, 1)
        plotbereich_layout_grid.addWidget(self.y_spinbox, 4, 1)
        self.Range_spinbox = QSpinBox(self)
        self.Range_spinbox.setRange(0, (Pixel-1))  # Bereich angepasst
        self.Range_spinbox.setValue(int(Pixel/2))
        self.Range_spinbox.valueChanged.connect(self.BereichChanges)
        self.Range_spinbox.setToolTip("Sets the size of the plotrange.")
        plotbereich_layout_grid.addWidget(QLabel("Size:", self), 5, 0)
        plotbereich_layout_grid.addWidget(self.Range_spinbox, 6, 0)

        # Mittelwertbereich Groupbox
        mittelwert_groupbox = QGroupBox("Mean range", self)
        self.tab1.grid.addWidget(mittelwert_groupbox, 4, 0)
        mittelwert_layout = QVBoxLayout(mittelwert_groupbox)
        self.x_mittel_spinbox = QSpinBox(self)
        self.x_mittel_spinbox.setRange(0, (Pixel-1))  # Bereich angepasst
        self.x_mittel_spinbox.setValue(int(Pixel/2))
        self.x_mittel_spinbox.valueChanged.connect(self.MittelChanges)
        self.x_mittel_spinbox.setToolTip("Sets the X-Position of the centerpoint of the meanvalue area.")
        mittelwert_layout.addWidget(QLabel("X-Position:", self))
        mittelwert_layout.addWidget(self.x_mittel_spinbox)
        self.y_mittel_spinbox = QSpinBox(self)
        self.y_mittel_spinbox.setRange(0, (Pixel-1))  # Bereich angepasst
        self.y_mittel_spinbox.setValue(int(Pixel/2))
        self.y_mittel_spinbox.valueChanged.connect(self.MittelChanges)
        self.y_mittel_spinbox.setToolTip("Sets the Y-Position of the centerpoint of the meanvalue area.")
        mittelwert_layout.addWidget(QLabel("Y-Position:", self))
        mittelwert_layout.addWidget(self.y_mittel_spinbox)
        self.size_spinbox = QSpinBox(self)
        self.size_spinbox.setRange(1, Pixel)
        self.size_spinbox.setValue(20)
        self.size_spinbox.valueChanged.connect(self.MittelChanges)
        self.size_spinbox.setToolTip("Sets the size of the meanvalue area.")
        mittelwert_layout.addWidget(QLabel("Size:", self))
        mittelwert_layout.addWidget(self.size_spinbox)

        # Ordnerauswahl
        ordner_groupbox = QGroupBox("Path selection", self)
        self.tab1.grid.addWidget(ordner_groupbox, 5, 0, 1, 5)
        ordner_layout = QGridLayout(ordner_groupbox)
        self.ordner_label = QLabel("No path selected", self)
        self.ordner_button = QPushButton("Load path", self)
        self.ordner_button.clicked.connect(self.ordner_auswaehlen)
        self.ordner_button.setToolTip("Load the path, where your data is located.")
        ordner_layout.addWidget(self.ordner_button, 0, 0)
        ordner_layout.addWidget(self.ordner_label, 0, 1, 0, 4)

        # Widget für den Plot
        self.plot_groupbox1 = QGroupBox(self)
        self.tab1.grid.addWidget(self.plot_groupbox1, 0, 1, 5, 1)
        self.plot_layout1 = QGridLayout(self.plot_groupbox1)
        #self.plot_widget = QWidget(self)
        #self.plot_widget.setFixedSize(800,800)
        #self.plot_layout = QVBoxLayout(self.plot_widget)
        #self.plot_layout.setContentsMargins(0, 0, 0, 0)
        #self.plot_widget.setSizePolicy(QSizePolicy.Expanding,
        #                               QSizePolicy.Expanding)
        self.tab1.grid.addWidget(self.plot_groupbox1, 0, 1, 5, 1)

  
        
        self.tab1.setLayout(self.tab1.grid)   

        # Initialer Live-Plot
        self.initial_plot()

        #---------------------------- Tab2 --------------------------
        self.tab2.grid = QGridLayout()

        # Plots generieren Button
        self.plot_button2 = QPushButton("Generate plots", self)
        self.plot_button2.clicked.connect(self.plots_generieren_spectra)
        self.plot_button2.setToolTip("Start plotting Spectra.")                                                   #Setzt eine Buttonbeschreibung bei MouseOver
        self.plot_button2.setStyleSheet("color: black; background-color: rgb(0,255,0); border-radius: 10px")
        self.plot_button2.setFixedHeight(50)
        self.plot_button2.setFixedWidth(320)
        self.tab2.grid.addWidget(self.plot_button2, 0, 0)

        # Overview generieren
        self.load_button = QPushButton('Load Overview')
        self.load_button.setStyleSheet("color: white; background-color: rgb(0,0,255); border-radius: 10px")
        self.load_button.setFixedHeight(50)
        self.load_button.setFixedWidth(320)
        self.load_button.clicked.connect(self.load_data)
        self.load_button.setToolTip("Load an overview image to select the pixel to plot.")          
        self.tab2.grid.addWidget(self.load_button, 1, 0)

        # Datarange Groupbox
        limits_groupbox2 = QGroupBox("Limits [OD]", self)
        self.tab2.grid.addWidget(limits_groupbox2, 2, 0)
        limits_layout2 = QGridLayout(limits_groupbox2)
        self.upper_spinbox2 = QDoubleSpinBox(self)
        self.upper_spinbox2.setRange(-100, 100)  # Bereich angepasst
        self.upper_spinbox2.setValue(2)
        self.upper_spinbox2.setSingleStep(0.01)
        self.upper_spinbox2.setToolTip("Sets the upper limits for the plots")    
        #self.upper_spinbox2.valueChanged.connect(self.SomeChanges)
        self.lower_spinbox2 = QDoubleSpinBox(self)
        self.lower_spinbox2.setRange(-100, 100)  # Bereich angepasst
        self.lower_spinbox2.setValue(0)
        self.lower_spinbox2.setSingleStep(0.01)
        self.lower_spinbox2.setToolTip("Sets the lower limits for the plots")   
        #self.lower_spinbox2.valueChanged.connect(self.SomeChanges)
        limits_layout2.addWidget(QLabel("Lower limit:", self),0,0)
        limits_layout2.addWidget(self.lower_spinbox2,1,0)
        limits_layout2.addWidget(QLabel("Upper limit:", self),0,1)
        limits_layout2.addWidget(self.upper_spinbox2,1,1)
        self.colorbar_check2 = QCheckBox("Legend", self)
        self.colorbar_check2.setChecked(True)
        self.colorbar_check2.setToolTip("If checked: All plots will have a legend.")
        #self.colorbar_check.stateChanged.connect(self.SomeChanges)
        limits_layout2.addWidget(self.colorbar_check2,2,0)

        # Plotbereich Groupbox
        plotbereich_groupbox2 = QGroupBox("Plotrange", self)
        self.tab2.grid.addWidget(plotbereich_groupbox2, 3, 0)
        plotbereich_layout2 = QGridLayout(plotbereich_groupbox2)
        self.WL_spinbox_low = QSpinBox(self)
        self.WL_spinbox_low.setRange(350, 1000)  # Bereich angepasst
        self.WL_spinbox_low.setValue(400)
        self.WL_spinbox_low.setToolTip("Set the lower limit for the Wavalength.")
        #self.WL_spinbox_low.valueChanged.connect(self.BereichChanges)
        plotbereich_layout2.addWidget(QLabel("Wavelength min:", self),0,0)
        plotbereich_layout2.addWidget(self.WL_spinbox_low,1,0)
        self.WL_spinbox_high = QSpinBox(self)
        self.WL_spinbox_high.setRange(350, 1000)  # Bereich angepasst
        self.WL_spinbox_high.setValue(500)
        self.WL_spinbox_high.setToolTip("Set the upper limit for the Wavalength.")
        #self.WL_spinbox_high.valueChanged.connect(self.BereichChanges)
        plotbereich_layout2.addWidget(QLabel("Wavelength max:", self),0,1)
        plotbereich_layout2.addWidget(self.WL_spinbox_high,1,1)

        # Mittelwertbereich Groupbox
        mittelwert_groupbox2 = QGroupBox("Mean range", self)
        self.tab2.grid.addWidget(mittelwert_groupbox2, 4, 0)
        mittelwert_layout2 = QGridLayout(mittelwert_groupbox2)
        self.x_mittel_spinbox2 = QSpinBox(self)
        self.x_mittel_spinbox2.setRange(1, 100)  # Bereich angepasst
        self.x_mittel_spinbox2.setValue(4)
        self.x_mittel_spinbox2.setToolTip("Sets the number of pixels in X-Direction to average over.")
        #self.x_mittel_spinbox2.valueChanged.connect(self.MittelChanges)
        mittelwert_layout2.addWidget(QLabel("X-Range:", self),0,0)
        mittelwert_layout2.addWidget(self.x_mittel_spinbox2,1,0)
        self.y_mittel_spinbox2 = QSpinBox(self)
        self.y_mittel_spinbox2.setRange(1, 100)  # Bereich angepasst
        self.y_mittel_spinbox2.setValue(4)
        self.y_mittel_spinbox2.setToolTip("Sets the number of pixels in Y-Direction to average over.")
        #self.y_mittel_spinbox.valueChanged.connect(self.MittelChanges)
        mittelwert_layout2.addWidget(QLabel("Y-Range:", self),0,1)
        mittelwert_layout2.addWidget(self.y_mittel_spinbox2,1,1)

        #SavGol-Filter
        self.groupboxSavGol2 = QGroupBox("Use Savitzky-Golay-Filter", self) 
        self.groupboxSavGol2.setCheckable(True)
        self.groupboxSavGol2.setChecked(True)
        self.groupboxSavGol2.setToolTip("Use Savitzky-Golay-Filter to smooth the Data")
        self.tab2.grid.addWidget(self.groupboxSavGol2, 5, 0)
        #limits_layout = QVBoxLayout(limits_groupbox)
        SavGol_layout_grid2 = QGridLayout(self.groupboxSavGol2)
        self.SGArea_spinbox2 = QSpinBox(self)
        self.SGArea_spinbox2.setRange(0, 1000000)  # Bereich angepasst
        self.SGArea_spinbox2.setValue(20)
        #self.SGArea_spinbox2.valueChanged.connect(self.BereichChanges)
        self.SGArea_spinbox2.setToolTip("Set the upper Limit for the integrated plots.")
        self.SGPoly_spinbox2 = QSpinBox(self)
        self.SGPoly_spinbox2.setRange(0, 1000000)  # Bereich angepasst
        self.SGPoly_spinbox2.setValue(3)
        #self.SGPoly_spinbox2.valueChanged.connect(self.BereichChanges)
        self.SGPoly_spinbox2.setToolTip("Set the lower Limit for the integrated plots.")
        self.UseBoth_check2 = QCheckBox("Plot Original Data", self)
        self.UseBoth_check2.setChecked(False)
        #self.UseBoth_check2.stateChanged.connect(self.BereichChanges)
        self.UseBoth_check2.setToolTip("If checked: Original- and Savitzky-Golay-Filter-Data will be plotted.")
        self.ShowMax_check2 = QCheckBox("Display Max.", self)
        self.ShowMax_check2.setChecked(False)
        #self.ShowMax_check2.stateChanged.connect(self.BereichChanges)
        self.ShowMax_check2.setToolTip("If checked: All 2D plots will have display of the maximum values.")
        self.ShowMin_check2 = QCheckBox("Display Min.", self)
        self.ShowMin_check2.setChecked(False)
        #self.ShowMin_check2.stateChanged.connect(self.BereichChanges)
        self.ShowMin_check2.setToolTip("If checked: All 2D plots will have display of the min values.")

        SavGol_layout_grid2.addWidget(QLabel("SG Area:", self), 0, 1)
        SavGol_layout_grid2.addWidget(QLabel("SG Polynom:", self), 0, 0)

        SavGol_layout_grid2.addWidget(self.SGArea_spinbox2, 1, 1)
        SavGol_layout_grid2.addWidget(self.SGPoly_spinbox2, 1, 0)
        SavGol_layout_grid2.addWidget(self.UseBoth_check2, 3, 0)
        SavGol_layout_grid2.addWidget(self.ShowMax_check2, 2, 0)
        SavGol_layout_grid2.addWidget(self.ShowMin_check2, 2, 1)

        #Inset
        self.groupboxInset2 = QGroupBox("Plot inset", self) 
        self.groupboxInset2.setCheckable(True)
        self.groupboxInset2.setChecked(False)
        self.groupboxInset2.setToolTip("Plot inset of interesting area")
        self.tab2.grid.addWidget(self.groupboxInset2, 6, 0)
        #limits_layout = QVBoxLayout(limits_groupbox)
        Inset_layout_grid2 = QGridLayout(self.groupboxInset2)
        self.InsetLowX_spinbox2 = QSpinBox(self)
        self.InsetLowX_spinbox2.setRange(0, 1000000)  # Bereich angepasst
        self.InsetLowX_spinbox2.setValue(400)
        self.InsetLowX_spinbox2.setToolTip("Set the upper X-Limit for the integrated plots.")
        self.InsetHighX_spinbox2 = QSpinBox(self)
        self.InsetHighX_spinbox2.setRange(0, 1000000)  # Bereich angepasst
        self.InsetHighX_spinbox2.setValue(500)
        self.InsetHighX_spinbox2.setToolTip("Set the lower X-Limit for the integrated plots.")
        self.InsetLowY_spinbox2 = QSpinBox(self)
        self.InsetLowY_spinbox2.setRange(0, 1000000)  # Bereich angepasst
        self.InsetLowY_spinbox2.setValue(400)
        #self.InsetLowY_spinbox2.valueChanged.connect(self.BereichChanges)
        self.InsetLowY_spinbox2.setToolTip("Set the upper Y-Limit for the integrated plots.")
        self.InsetHighY_spinbox2 = QSpinBox(self)
        self.InsetHighY_spinbox2.setRange(0, 1000000)  # Bereich angepasst
        self.InsetHighY_spinbox2.setValue(500)
        #self.InsetHighY_spinbox2.valueChanged.connect(self.BereichChanges)
        self.InsetHighY_spinbox2.setToolTip("Set the lower Y-Limit for the integrated plots.")
        Inset_layout_grid2.addWidget(QLabel("Lower X-Limit:", self), 0, 0)
        Inset_layout_grid2.addWidget(QLabel("Upper X-Limit:", self), 0, 1)
        Inset_layout_grid2.addWidget(self.InsetLowX_spinbox2, 1, 0)
        Inset_layout_grid2.addWidget(self.InsetHighX_spinbox2, 1, 1)
        Inset_layout_grid2.addWidget(QLabel("Lower Y-Limit:", self), 2, 0)
        Inset_layout_grid2.addWidget(QLabel("Upper Y-Limit:", self), 2, 1)
        Inset_layout_grid2.addWidget(self.InsetLowY_spinbox2, 3, 0)
        Inset_layout_grid2.addWidget(self.InsetHighY_spinbox2, 3, 1)

        # Ordnerauswahl
        ordner_groupbox2 = QGroupBox("Path selection", self)
        self.tab2.grid.addWidget(ordner_groupbox2, 7, 0, 1, 5)
        ordner_layout2 = QGridLayout(ordner_groupbox2)
        self.ordner_label2 = QLabel("No path selected", self)
        self.ordner_button2 = QPushButton("Load path", self)
        self.ordner_button2.clicked.connect(self.ordner_auswaehlen_spectra)
        self.ordner_button2.setToolTip("Load the path, where your data is located.")
        ordner_layout2.addWidget(self.ordner_button2, 0, 0)
        ordner_layout2.addWidget(self.ordner_label2, 0, 1, 0, 4)

#--------------------------------------------------------------------------------------------------
        #Plot und Punktauswahl
        plot_groupbox2 = QGroupBox(self)
        self.tab2.grid.addWidget(plot_groupbox2, 0, 1, 7, 1)
        plot_layout2 = QGridLayout(plot_groupbox2)

        # --- Matplotlib Figure und Canvas ---
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        # self.figure.subplots_adjust(bottom=0.2) # Platz schaffen falls nötig
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Load Overview-File", color="white")
        self.ax.set_facecolor(((53/255),(53/255),(53/255)))
        self.figure.patch.set_facecolor(((53/255),(53/255),(53/255)))
        self.ax.grid(True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # --- Matplotlib Toolbar ---
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Textfeld für Klick-Koordinaten ---
        self.points_label = QLabel("Selected positions (X, Y):")
        self.points_display = QTextEdit()
        self.points_display.setReadOnly(True)
        self.points_display.setFixedHeight(100) # Feste Höhe für das Textfeld
        self.points_display.setToolTip("Shows all selected positions.")

        self.clear_button = QPushButton('Delete positions')
        self.clear_button.setStyleSheet("color: white; background-color: rgb(255,80,100); border-radius: 10px")
        self.clear_button.setFixedWidth(130)
        self.clear_button.setFixedHeight(30)
        self.clear_button.setToolTip("Deletes all selected positions.")
        self.clear_button.clicked.connect(self.clear_points)


        plot_layout2.addWidget(self.toolbar, 0, 0)
        plot_layout2.addWidget(self.clear_button, 0, 3)
        plot_layout2.addWidget(self.canvas, 1, 0, 6, 5)
        plot_layout2.addWidget(self.points_label, 7, 0)
        plot_layout2.addWidget(self.points_display, 8, 0, 1, 4)
        #plot_layout2.addWidget(self.load_button, 8, 0, 1, 2)

        self.tab2.grid.addWidget(plot_groupbox2, 0, 1, 5, 1)

#--------------------------------------------------------------------------------------------------

        self.tab2.setLayout(self.tab2.grid)   


        
        #------------------------ Tab3 -----------------------------
        self.tab3.grid = QGridLayout()

        # Plots generieren Button
        self.plot_button3 = QPushButton("Generate plots", self)
        self.plot_button3.clicked.connect(self.plots_generieren_spectralMap)
        self.plot_button3.setToolTip("Start plotting Spectra.")                                                   #Setzt eine Buttonbeschreibung bei MouseOver
        self.plot_button3.setStyleSheet("color: black; background-color: rgb(0,255,0); border-radius: 10px")
        self.plot_button3.setFixedHeight(50)
        self.plot_button3.setFixedWidth(320)
        self.tab3.grid.addWidget(self.plot_button3, 0, 0)

        # Overview generieren
        self.load_button3 = QPushButton('Generate Spectral Map')
        self.load_button3.setStyleSheet("color: white; background-color: rgb(0,0,255); border-radius: 10px")
        self.load_button3.setFixedHeight(50)
        self.load_button3.setFixedWidth(320)
        self.load_button3.clicked.connect(self.SpectralMapping)
        self.load_button3.setToolTip("Load an overview image to select the pixel to plot.")          
        self.tab3.grid.addWidget(self.load_button3, 1, 0)

        self.groupboxSavGol = QGroupBox("Use Savitzky-Golay-Filter", self) 
        self.groupboxSavGol.setCheckable(True)
        self.groupboxSavGol.setChecked(True)
        self.groupboxSavGol.setToolTip("Use Savitzky-Golay-Filter to smooth the Data")
        self.tab3.grid.addWidget(self.groupboxSavGol, 2, 0)
        #limits_layout = QVBoxLayout(limits_groupbox)
        SavGol_layout_grid3 = QGridLayout(self.groupboxSavGol)
        self.SGArea_spinbox = QSpinBox(self)
        self.SGArea_spinbox.setRange(0, 1000000)  # Bereich angepasst
        self.SGArea_spinbox.setValue(20)
        #self.SGArea_spinbox.valueChanged.connect(self.BereichChanges)
        self.SGArea_spinbox.setToolTip("Set the upper Limit for the integrated plots.")
        self.SGPoly_spinbox = QSpinBox(self)
        self.SGPoly_spinbox.setRange(0, 1000000)  # Bereich angepasst
        self.SGPoly_spinbox.setValue(3)
        #self.SGPoly_spinbox.valueChanged.connect(self.BereichChanges)
        self.SGPoly_spinbox.setToolTip("Set the lower Limit for the integrated plots.")
        self.UseBoth_check = QCheckBox("Plot Original Data", self)
        self.UseBoth_check.setChecked(False)
        #self.UseBoth_check.stateChanged.connect(self.BereichChanges)
        self.UseBoth_check.setToolTip("If checked: Original- and Savitzky-Golay-Filter-Data will be plotted.")

        SavGol_layout_grid3.addWidget(QLabel("SG Area:", self), 0, 1)
        SavGol_layout_grid3.addWidget(QLabel("SG Polynom:", self), 0, 0)

        SavGol_layout_grid3.addWidget(self.SGArea_spinbox, 1, 1)
        SavGol_layout_grid3.addWidget(self.SGPoly_spinbox, 1, 0)
        SavGol_layout_grid3.addWidget(self.UseBoth_check, 2, 0)

        # Datarange Groupbox
        limits_groupbox3 = QGroupBox("Limits", self)
        self.tab3.grid.addWidget(limits_groupbox3, 3, 0)
        #limits_layout = QVBoxLayout(limits_groupbox)
        limits_layout_grid3 = QGridLayout(limits_groupbox3)

        self.upper_spinbox3 = QSpinBox(self)
        self.upper_spinbox3.setRange(0, 1000000)  # Bereich angepasst
        self.upper_spinbox3.setValue(1000)
        #self.upper_spinbox3.valueChanged.connect(self.BereichChanges)
        self.upper_spinbox3.setToolTip("Set the upper Limit for the integrated plots.")
        self.lower_spinbox3 = QSpinBox(self)
        self.lower_spinbox3.setRange(0, 1000000)  # Bereich angepasst
        self.lower_spinbox3.setValue(0)
        #self.lower_spinbox3.valueChanged.connect(self.BereichChanges)
        self.lower_spinbox3.setToolTip("Set the lower Limit for the integrated plots.")

        limits_layout_grid3.addWidget(QLabel("Upper limit [Counts]:", self), 0, 1)
        limits_layout_grid3.addWidget(QLabel("Lower limit [Counts]:", self), 0, 0)

        limits_layout_grid3.addWidget(self.upper_spinbox3, 1, 1)
        limits_layout_grid3.addWidget(self.lower_spinbox3, 1, 0)

        self.upper_spinbox_Init3 = QSpinBox(self)
        self.upper_spinbox_Init3.setRange(300, 1200)  # Bereich angepasst
        self.upper_spinbox_Init3.setValue(500)
        #self.upper_spinbox_Init3.valueChanged.connect(self.BereichChanges)
        self.upper_spinbox_Init3.setToolTip("Set the upper Limit for the max. wavelength plots.")
        self.lower_spinbox_Init3 = QSpinBox(self)
        self.lower_spinbox_Init3.setRange(300, 1200)  # Bereich angepasst
        self.lower_spinbox_Init3.setValue(400)
        #self.lower_spinbox_Init3.valueChanged.connect(self.BereichChanges)
        self.lower_spinbox_Init3.setToolTip("Set the lower Limit for the max. wavelength plots.")

        limits_layout_grid3.addWidget(QLabel("Upper WL limit [nm]:", self), 2, 1)
        limits_layout_grid3.addWidget(QLabel("Lower WL limit [nm]:", self), 2, 0)
        limits_layout_grid3.addWidget(self.upper_spinbox_Init3, 3, 1)
        limits_layout_grid3.addWidget(self.lower_spinbox_Init3, 3, 0)


        self.limit3dInit_check3 = QCheckBox("Fixed limits 3D", self)
        self.limit3dInit_check3.setChecked(True)
        #self.limit3dInit_check3.stateChanged.connect(self.BereichChanges)
        self.limit3dInit_check3.setToolTip("If checked: All normal plots will have fixed upper and lower limits.")
        self.limits3d_check3 = QCheckBox("Fixed WL limits 3D", self)
        self.limits3d_check3.setChecked(True)
        #self.limits3d_check3.stateChanged.connect(self.BereichChanges)
        self.limits3d_check3.setToolTip("If checked: All 3D absorption plots will have fixed upper and lower limits.")
        self.limits2d_check3 = QCheckBox("Fixed WL limits 2D", self)
        self.limits2d_check3.setChecked(True)
        #self.limits2d_check3.stateChanged.connect(self.BereichChanges)
        self.limits2d_check3.setToolTip("If checked: All 2D absorption plots will have fixed upper and lower limits.")
        self.colorbar_check3 = QCheckBox("Colorbar", self)
        self.colorbar_check3.setChecked(True)
        #self.colorbar_check3.stateChanged.connect(self.BereichChanges)
        self.colorbar_check3.setToolTip("If checked: All 3D plots will have a colorbar.")
        self.SVG_check3 = QCheckBox("Save as SVG", self)
        self.SVG_check3.setChecked(False)
        #self.SVG_check3.stateChanged.connect(self.BereichChanges)
        self.SVG_check3.setToolTip("If checked: all plots will be saved as svg-Files.")
        self.PixelPics_check3 = QCheckBox("Save all Pixel", self)
        self.PixelPics_check3.setChecked(False)
        #self.PixelPics_check3.stateChanged.connect(self.BereichChanges)
        self.PixelPics_check3.setToolTip("If checked: All pixel data will be plotted. This will take a wile.")

        limits_layout_grid3.addWidget(self.limits2d_check3, 4, 0)
        limits_layout_grid3.addWidget(self.limits3d_check3, 4, 1)
        limits_layout_grid3.addWidget(self.limit3dInit_check3, 5, 0)
        limits_layout_grid3.addWidget(self.PixelPics_check3, 5, 1)
        limits_layout_grid3.addWidget(self.colorbar_check3, 6, 0)
        limits_layout_grid3.addWidget(self.SVG_check3, 6, 1)

        # Plotbereich Groupbox
        plotbereich_groupbox3 = QGroupBox("Plotrange 2D", self)
        self.tab3.grid.addWidget(plotbereich_groupbox3, 4, 0)
        #plotbereich_layout3 = QVBoxLayout(plotbereich_groupbox3)
        plotbereich_layout3 = QGridLayout(plotbereich_groupbox3)
        self.WL_spinbox_low3 = QSpinBox(self)
        self.WL_spinbox_low3.setRange(350, 1000)  # Bereich angepasst
        self.WL_spinbox_low3.setValue(400)
        self.WL_spinbox_low3.setToolTip("Set the lower limit for the Wavalength.")
        #self.WL_spinbox_low.valueChanged.connect(self.BereichChanges)
        plotbereich_layout3.addWidget(QLabel("Wavelength min:", self), 0, 0)
        plotbereich_layout3.addWidget(self.WL_spinbox_low3, 1, 0)
        self.WL_spinbox_high3 = QSpinBox(self)
        self.WL_spinbox_high3.setRange(350, 1000)  # Bereich angepasst
        self.WL_spinbox_high3.setValue(500)
        self.WL_spinbox_high3.setToolTip("Set the upper limit for the Wavalength.")
        #self.WL_spinbox_high.valueChanged.connect(self.BereichChanges)
        self.SVG_check2D3 = QCheckBox("Save as SVG", self)
        self.SVG_check2D3.setChecked(False)
        #self.SVG_check2D3.stateChanged.connect(self.BereichChanges)
        self.SVG_check2D3.setToolTip("If checked: all plots will be saved as svg-Files.")
        self.PixelPics_check2D3 = QCheckBox("Save all Pixel", self)
        self.PixelPics_check2D3.setChecked(False)
        #self.PixelPics_check3.stateChanged.connect(self.BereichChanges)
        self.PixelPics_check2D3.setToolTip("If checked: All pixel data will be plotted. This will take a wile.")
        self.legend_check2D3 = QCheckBox("Legend", self)
        self.legend_check2D3.setChecked(True)
        #self.legend_check2D3.stateChanged.connect(self.BereichChanges)
        self.legend_check2D3.setToolTip("If checked: All 2D plots will have a legend.")
        self.ShowMax_check3 = QCheckBox("Display Max.", self)
        self.ShowMax_check3.setChecked(True)
        self.ShowMax_check3.setToolTip("If checked: All 2D plots will have display of the maximum value.")
        plotbereich_layout3.addWidget(QLabel("Wavelength max:", self), 0, 1)
        plotbereich_layout3.addWidget(self.WL_spinbox_high3, 1, 1)

        self.upper_spinbox32 = QSpinBox(self)
        self.upper_spinbox32.setRange(0, 1000000)  # Bereich angepasst
        self.upper_spinbox32.setValue(1000)
        #self.upper_spinbox3.valueChanged.connect(self.BereichChanges)
        self.upper_spinbox32.setToolTip("Set the upper Limit for the integrated plots.")
        self.lower_spinbox32 = QSpinBox(self)
        self.lower_spinbox32.setRange(0, 1000000)  # Bereich angepasst
        self.lower_spinbox32.setValue(0)
        #self.lower_spinbox3.valueChanged.connect(self.BereichChanges)
        self.lower_spinbox32.setToolTip("Set the lower Limit for the integrated plots.")

        plotbereich_layout3.addWidget(QLabel("Upper limit [Counts]:", self), 2, 1)
        plotbereich_layout3.addWidget(QLabel("Lower limit [Counts]:", self), 2, 0)

        plotbereich_layout3.addWidget(self.upper_spinbox32, 3, 1)
        plotbereich_layout3.addWidget(self.lower_spinbox32, 3, 0)


        plotbereich_layout3.addWidget(self.legend_check2D3, 4, 0)
        plotbereich_layout3.addWidget(self.ShowMax_check3, 4, 1)
        plotbereich_layout3.addWidget(self.SVG_check2D3, 5, 0)
        plotbereich_layout3.addWidget(self.PixelPics_check2D3, 5, 1)

        """
        Colors_groupbox3 = QGroupBox("Colormap selection", self)
        self.tab3.grid.addWidget(Colors_groupbox3, 5, 0)
        Color_layout3 = QGridLayout(Colors_groupbox3)
        self.PlotColors1 = QComboBox(self)
        #self.PlotColors1.setFixedSize(150, 30)          
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Autumn.png"), "autumn")    
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/BanksyCMAP.png"), "BanksyCMAP")          
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Blues.png"), "Blues")   
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Bone.png"), "bone")           
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/BrBG.png"), "BrBG")                  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/brg.png"), "brg")        
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/BuGn.png"), "BuGn")         
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/BuPu.png"), "BuPu")                      
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/bwr.png"), "bwr")  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/cividis.png"), "cividis")  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Cool.png"), "cool")   
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/CMRmap.png"), "CMRmap")          
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/coolwarm.png"), "coolwarm") 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Copper.png"), "copper")  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/cubehelix.png"), "cubehelix")        
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/gist_earth.png"), "gist_earth")
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Heat.png"), "gist_heat") 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/gist_ncar.png"), "gist_ncar")                         
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/gist_rainbow.png"), "gist_rainbow")                 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/gist_stern.png"), "gist_stern")              
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/gnuplot.png"), "gnuplot")         
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/gnuplot2.png"), "gnuplot2")   
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Grey.png"), "gray")          
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Hot.png"), "hot")              
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/hsv.png"), "hsv")  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/HydraCMAP2.png"), "HydraCMAP2")
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/HydraCMAP.png"), "HydraCMAP")
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Inferno.png"), "inferno")           
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/jet.png"), "jet") 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Magma.png"), "magma")
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/nipy_spectral.png"), "nipy_spectral")         
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/PiYG.png"), "PiYG") 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Plasma.png"), "plasma")   
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/PuBu.png"), "PuBu")     
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/PuOr.png"), "PuOr")     
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/PuRd.png"), "PuRd")              
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/RdGy.png"), "RdGy")   
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/RdYlBu.png"), "RdYlBu")       
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/RdYlGn.png"), "RdYlGn")     
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Reds.png"), "Reds") 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/rhkCMAP.png"), "rhkCMAP")
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Spectral.png"), "Spectral")     
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Spring.png"), "spring")     
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Summer.png"), "summer") 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/terrain.png"), "terrain") 
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Viridis.png"), "viridis")
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Winter.png"), "winter")  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/Wistia.png"), "Wistia")  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/WsxmCMAP.png"), "wsxmCMAP")       
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/YlOrBr.png"), "YlOrBr")  
        self.PlotColors1.addItem(QIcon("/home/pi/Desktop/HydraScan/Files/Styles/YlOrRd.png"), "YlOrRd")   
        self.PlotColors1.setCurrentIndex(45)

        self.InvertCMAPLive1 = QCheckBox("Invert Color", self)

        Color_layout3.addWidget(self.PlotColors1, 0, 0)
        Color_layout3.addWidget(self.InvertCMAPLive1, 1, 0)
        """

        # Ordnerauswahl
        ordner_groupbox3 = QGroupBox("File selection", self)
        self.tab3.grid.addWidget(ordner_groupbox3, 6, 0, 1, 5)
        ordner_layout3 = QGridLayout(ordner_groupbox3)
        self.ordner_label3 = QLabel("No file selected", self)
        self.ordner_button3 = QPushButton("Load File", self)
        self.ordner_button3.clicked.connect(self.load_data_SpectralMap)
        self.ordner_button3.setToolTip("Load the file, containing your data.")
        ordner_layout3.addWidget(self.ordner_button3, 0, 0)
        ordner_layout3.addWidget(self.ordner_label3, 0, 1, 0, 4)
        
        plot_groupbox3 = QGroupBox(self)
        self.tab3.grid.addWidget(plot_groupbox3, 0, 1, 5, 1)
        plot_layout3 = QGridLayout(plot_groupbox3)

        # --- Matplotlib Figure und Canvas ---
        self.figure3 = Figure(figsize=(5, 5), dpi=100)
        self.canvas3 = FigureCanvas(self.figure3)
        # self.figure.subplots_adjust(bottom=0.2) # Platz schaffen falls nötig
        self.ax3 = self.figure3.add_subplot(111)
        self.ax3.set_title("Load ASC-File", color="white")
        self.ax3.set_facecolor(((53/255),(53/255),(53/255)))
        self.figure3.patch.set_facecolor(((53/255),(53/255),(53/255)))
        self.ax3.grid(True)
        self.canvas3.mpl_connect('button_press_event', self.on_click3)

        # --- Matplotlib Toolbar ---
        self.toolbar3 = NavigationToolbar(self.canvas3, self)

        # --- Textfeld für Klick-Koordinaten ---
        self.points_label3 = QLabel("Selected positions (X, Y):")
        self.points_display3 = QTextEdit()
        self.points_display3.setReadOnly(True)
        self.points_display3.setFixedHeight(100) # Feste Höhe für das Textfeld
        self.points_display3.setToolTip("Shows all selected positions.")

        self.clear_button3 = QPushButton('Delete positions')
        self.clear_button3.setStyleSheet("color: white; background-color: rgb(255,80,100); border-radius: 10px")
        self.clear_button3.setFixedWidth(130)
        self.clear_button3.setFixedHeight(30)
        self.clear_button3.setToolTip("Deletes all selected positions.")
        self.clear_button3.clicked.connect(self.clear_points3)

        plot_layout3.addWidget(self.toolbar3, 0, 0)
        plot_layout3.addWidget(self.clear_button3, 0, 3)
        plot_layout3.addWidget(self.canvas3, 1, 0, 6, 4)
        plot_layout3.addWidget(self.points_label3, 7, 0)
        plot_layout3.addWidget(self.points_display3, 8, 0, 1, 4)


        self.tab3.setLayout(self.tab3.grid) 


        #-------------- Tabs zum Widget hinzufügen ------------------
        global CITE
        self.labelCite = QLabel("If HydraAbsorption contributes to publisch a work please cite:\t" + CITE, self)                                                                                              #setzt ein Label    
        self.labelCite.setFont(QFont('Arial', 8))

        self.layout.addWidget(self.tabs)                                                                                                        #Fügt die Tabs zum Layout hinzu
        self.layout.addWidget(self.labelCite)
        self.setLayout(self.layout)            

        self.show()


    def load_data_SpectralMap(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                          "Load ASC-Files",
                                                          "", # Startverzeichnis
                                                          "ASC (*.asc);;All Files (*)",
                                                          options=options)
        if fileName:
            self.ordner_label3.setText(f"File: {fileName}")


    def SpectralMapping(self):
        ordnerpfadSpectralMap = self.ordner_label3.text().split(" ")[-1]
        print(ordnerpfadSpectralMap)
        if not os.path.isfile(ordnerpfadSpectralMap):
            QMessageBox.warning(self, "Error", "No valid file selected!")
            return
        self.path_spectralmap = ordnerpfadSpectralMap

        SavGol_Area=self.SGArea_spinbox.value()
        SavGol_Polynom=self.SGPoly_spinbox.value()

        self.readdata_SpectralMap(SavGol_Area, SavGol_Polynom)
        self.MatrizenBerechnen()
        self.PlotSpectralMap()
        self.PlotWeightedSpectralMap()

    def readdata_SpectralMap(self,SavGol_Area, SavGol_Polynom):
        data = np.loadtxt(open(self.path_spectralmap,'rt').readlines()[:-1], dtype=None)

        x_axis_values = data[:, 0]
        self.WLsSpectralMap = x_axis_values
        pixel_data = data[:, 1:]
        self.originalSM = pixel_data
        self.TransposedData = np.transpose(pixel_data)

        # Dimensionen prüfen
        num_x_points = pixel_data.shape[0]  # Anzahl der Zeilen (entspricht X-Punkten)
        self.num_pixel_indices = pixel_data.shape[1] # Anzahl der Spalten (entspricht Pixeln)
        print(f"Daten geladen: {num_x_points} Wellenlängen, {self.num_pixel_indices} Spektren.")

        x = int(round(math.sqrt(self.num_pixel_indices)))
        y = int(round(math.sqrt(self.num_pixel_indices)))
        
        
        # Integration durchführen
        self.integrated_matrix = np.sum(pixel_data, axis=0)
        print(f"'Integrierte' Matrix (kumulative Summe entlang Achse 0) erstellt mit Shape: {self.integrated_matrix.shape}")

        #SavGol-Filter anwenden
        pixel_data_savgol = savgol_filter(pixel_data, SavGol_Area, SavGol_Polynom, axis=0)
        self.pixel_data_savgol_SM = pixel_data_savgol
        max_positions = np.argmax(pixel_data_savgol, axis=0)
        print(f"Positionen (Indizes entlang X-Achse) der Maxima pro Pixel berechnet. Shape: {max_positions.shape}")

        i = 0
        savgol_matrix_WL = []
        while i < len(max_positions):
            savgol_matrix_WL.append(x_axis_values[max_positions[i]])
            i+=1
        self.savgol_matrix = np.array(savgol_matrix_WL, dtype=np.float32)
        print(f"Wellenlängen den Maxima pro Pixel zugeordnet. Shape: {self.savgol_matrix.shape}")

        print("daten vollständig eingelesen")

    def MatrizenBerechnen(self):
        self.pixel_matrix = self.integrated_matrix.reshape((int(round(math.sqrt(self.num_pixel_indices))), int(round(math.sqrt(self.num_pixel_indices)))))
        print(f"Pixel-Daten erfolgreich als {self.pixel_matrix.shape} Matrix interpretiert.")
        self.pixel_matrix_savgol = self.savgol_matrix.reshape((int(round(math.sqrt(self.num_pixel_indices))), int(round(math.sqrt(self.num_pixel_indices)))))
        print(f"Pixel-Daten erfolgreich als {self.pixel_matrix_savgol.shape} Matrix interpretiert.")
        self.originalSM_matrix = self.originalSM.reshape((int(round(math.sqrt(self.num_pixel_indices))), int(round(math.sqrt(self.num_pixel_indices))),1023))
        print(f"Pixel-Daten erfolgreich als {self.originalSM_matrix.shape} Matrix interpretiert.")
        self.Transposed_originalSM_matrix = self.TransposedData.reshape((int(round(math.sqrt(self.num_pixel_indices))), int(round(math.sqrt(self.num_pixel_indices))),1023))
        print(f"Pixel-Daten erfolgreich als {self.originalSM_matrix.shape} Matrix interpretiert.")
        self.integrated_originalSM_matrix = np.sum(self.originalSM_matrix, axis=2)

    def PlotSpectralMap(self, cmap = "viridis"):
        #Übergabe an GUI
        self.ax3
        self.z_data3 = self.pixel_matrix_savgol

        # Gitterkoordinaten (X, Y) erstellen (basierend auf Indizes)
        ny, nx = self.z_data3.shape
        x = np.arange(nx + 1) # +1 für pcolormesh Kanten
        y = np.arange(ny + 1) # +1 für pcolormesh Kanten
        self.X3, self.Y3 = np.meshgrid(x, y)

        # Alten Plot löschen und neu zeichnen
        self.ax3.clear()
        # cmap='viridis' ist eine gute Standard-Colormap
        self.mesh_plot3 = self.ax3.pcolormesh(self.X3, self.Y3, self.z_data3,
                                            shading='flat', # oder 'auto'/'gouraud'
                                            cmap='Greys',
                                            snap=True) # Wichtig für genaue Klicks an Kanten
        #self.mesh_plot3 = self.ax3.pcolormesh(self.X3, self.Y3, self.pixel_matrix,
        #                                    shading='flat', # oder 'auto'/'gouraud'
        #                                    cmap='Greys',
        #                                    snap=True) # Wichtig für genaue Klicks an Kanten
        

        #self.figure.colorbar(self.mesh_plot, ax=self.ax, orientation='vertical')
        #self.figure.colorbar.set_fontcolor("white")
        #self.figure.colorbar(self.mesh_plot, ax=self.ax)
        #self.figure.colorbar.set_label(label="Z-Wert", color="white")
        #self.ax.set_xlabel("X Index")
        #self.ax.set_ylabel("Y Index")
        self.ax3.set_title("Click to select position!", color="white")
        self.ax3.set_facecolor(((53/255),(53/255),(53/255)))
        self.figure3.patch.set_facecolor(((53/255),(53/255),(53/255)))
        self.ax3.xaxis.label.set_color('white')
        self.ax3.yaxis.label.set_color('white')
        self.ax3.tick_params(axis='x', colors='white')
        self.ax3.tick_params(axis='y', colors='white')
        # Optional: Seitenverhältnis anpassen, falls die Zellen quadratisch sein sollen
        self.ax3.set_aspect('equal', adjustable='box')
        self.ax3.grid(True) # Gitter wieder hinzufügen
        self.canvas3.draw() # Wichtig: Canvas neu zeichnen

        # Geklickte Punkte zurücksetzen, wenn neue Daten geladen werden
        self.clear_points3() # Ruft auch update_points_display auf

        #Plotten der integrierten Matrix und der Max. Wellenlänge
        fig, axs = plt.subplots(figsize=(16, 7), nrows=1, ncols=2) # Größe des Plots anpassen

        im0 = axs[0].pcolormesh(self.pixel_matrix, cmap=cmap, shading='auto')
        im1 = axs[1].pcolormesh(self.pixel_matrix_savgol, cmap=cmap, shading='auto')

        #Farbleiste1 hinzufügen
        cbar0 = fig.colorbar(im0, ax=axs[0])
        cbar0.set_label('Integrierter Wert [Counts]')

        #Farbleiste2 hinzufügen
        cbar1 = fig.colorbar(im1, ax=axs[1])
        cbar1.set_label('Wavelength [nm]')

        #Achsenbeschriftungen und Titel
        axs[0].set_xlabel("X-Achse")
        axs[0].set_ylabel("Y-Achse")
        axs[0].set_title("Mesh Plot der 'integrierten' Daten")

        #Achsenbeschriftungen und Titel
        axs[1].set_xlabel("X-Achse")
        axs[1].set_ylabel("Y-Achse")
        axs[1].set_title("Mesh Plot der 'SavGol' Daten")

        #Layout optimieren und Plot anzeigen
        fig.tight_layout() # Verhindert Überlappung von Beschriftungen
        plt.show()

    def PlotWeightedSpectralMap(self, cmap = "viridis"):
        index_x = 0; index_y = 1; index_z = 2; index_c = 3
        list_name_variables = ['x', 'y', 'Intensity [Counts]', 'Wavelength [nm]']
        x = np.linspace(0, int(round(math.sqrt(self.num_pixel_indices))-1), int(round(math.sqrt(self.num_pixel_indices))))
        y = np.linspace(0, int(round(math.sqrt(self.num_pixel_indices))-1), int(round(math.sqrt(self.num_pixel_indices))))
        x2, y2 = np.meshgrid(x, y)

        #Coupeling the colormap with the Wavelengths
        color_dimension = self.pixel_matrix_savgol
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = matplotlib.colors.Normalize(minn, maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

        #3D-Surface Plot with extra dimension in Colorbar
        fig3D = plt.figure(); ax3D = fig3D.gca(projection='3d')
        surf = ax3D.plot_surface(x2,y2,self.pixel_matrix, facecolors = fcolors, linewidth=0, rstride=1, cstride=1,
                            antialiased=False)
        cbar = fig3D.colorbar(m, shrink=0.5, aspect=5)
        cbar.ax.get_yaxis().labelpad = 15; cbar.ax.set_ylabel(list_name_variables[index_c], rotation = 270)
        ax3D.set_xlabel(list_name_variables[index_x]); ax3D.set_ylabel(list_name_variables[index_y])
        ax3D.set_zlabel(list_name_variables[index_z])
        plt.title("3D-Surface of the integrated intensity with the max. wavelength as colormap")
        plt.show()

        #3D-Surface Plot with extra dimension in Colorbar
        #self.pixel_matrix_savgol_Normalized = np.interp(self.pixel_matrix_savgol, (self.pixel_matrix_savgol.min(), self.pixel_matrix_savgol.max()), (0, +1))
        #self.pixel_matrix_Normalized = np.interp(self.pixel_matrix, (self.pixel_matrix.min(), self.pixel_matrix.max()), (0, +1))
        self.pixel_matrix_Normalized = np.interp(self.pixel_matrix, (0, self.pixel_matrix.max()), (0, +1))
        #self.pixel_matrix_savgol_Normalized = np.interp(self.pixel_matrix_savgol, (0, self.pixel_matrix_savgol.max()), (0, +1))
        fig = plt.figure()

        #plt.style.use('dark_background')
        ax = fig.add_subplot(111)
        ax.set_facecolor(((0/255),(0/255),(0/255)))
        #mesh = ax.pcolormesh(x2,y2,self.pixel_matrix, cmap = cmap, alpha=self.pixel_matrix_savgol_Normalized ,vmin=InitMin, vmax=InitMax)
        mesh = ax.pcolormesh(x2,y2,self.pixel_matrix_savgol, cmap = cmap, alpha=self.pixel_matrix_Normalized ,vmin=self.pixel_matrix_savgol.min(), vmax=self.pixel_matrix_savgol.max())

        #Farbleiste1 hinzufügen
        cbar1 = fig.colorbar(mesh, ax=ax)
        cbar1.set_label('Wavelength [nm]', color='black')
        Ticks = np.arange(self.pixel_matrix_savgol.min(),self.pixel_matrix_savgol.max(),1, dtype=int)
        cbar1.set_ticklabels(Ticks,color='black')
        cbar1.outline.set_color("black")
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')

        #Achsenbeschriftungen und Titel
        ax.set_xlabel("X-Achse", color='black')
        ax.set_ylabel("Y-Achse", color='black')
        ax.set_title("Intensity weighted plot of the PL wavelength", color='black')
        plt.show()

    def Spectra_from_SpectralMap(self):
        ordnerpfad_spectralMap = self.path_spectralmap
        ordnerpfad_spectralMap = ordnerpfad_spectralMap[:ordnerpfad_spectralMap.rfind("/")+1]
        if not os.path.isdir(ordnerpfad_spectralMap):
            QMessageBox.warning(self, "Error", "No valid path selected!")
            return
        
        SpotsStart = self.clicked_points3
        NumberOfSamples = len(SpotsStart)

        if NumberOfSamples == 0:
            QMessageBox.warning(self, "Error", "No positions selected!")
            return

        Inthigh = self.upper_spinbox32.value()
        Intlow = self.lower_spinbox32.value()
        Legend = self.colorbar_check3.isChecked()
        WLhigh = self.WL_spinbox_high3.value()
        WLlow = self.WL_spinbox_low3.value()
        SavGol_Area = self.SGArea_spinbox.value()
        SavGol_Polynom = self.SGPoly_spinbox.value()
        ShowMax = self.ShowMax_check3.isChecked()
        SavGol = self.groupboxSavGol.isChecked()
        PlotBoth = self.UseBoth_check.isChecked()


        # Load Data
        WLs = self.WLsSpectralMap
        
        # Create Figure, Grid and Axis
        figSM = plt.figure(figsize=(8/2.54, 8/2.54), dpi=300, constrained_layout=True)
        gs = figSM.add_gridspec(1, 1)
        figSM_ax1 = figSM.add_subplot(gs[0, 0])

        ColorList = ['b','g','r','orange','c','m','y','k','lime','darorange','firebrick','hotpink','deepskyblue','crimson','pink','indigo','teal','limegreen','mediumvioletred','blueviolet','darkgoldenrod','lavender','navajowhite','saddlebrown','palevioletred','dodgerblue','lightslategrey']
        ColorListAlt = ['c','m','y','b','g','r','orange','k','lime','darorange','firebrick','hotpink','deepskyblue','crimson','pink','indigo','teal','limegreen','mediumvioletred','blueviolet','darkgoldenrod','lavender','navajowhite','saddlebrown','palevioletred','dodgerblue','lightslategrey']

        # Plotting and Axis Formatting fog1_ax1
        i = 0
        while i < NumberOfSamples:
                if SavGol == True and PlotBoth == True:
                    Label = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1]) + " SavGol" 
                    Label2 = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1])  
                    Data = savgol_filter(self.Transposed_originalSM_matrix[SpotsStart[i][1]][SpotsStart[i][0]], SavGol_Area, SavGol_Polynom)
                    Data2 = self.Transposed_originalSM_matrix[SpotsStart[i][1]][SpotsStart[i][0]]
                    figSM_ax1.plot(WLs, Data2, label=Label2, color=ColorList[i], linestyle='dashed')
                elif SavGol == True:
                    Label = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1]) + " SavGol"
                    Data = savgol_filter(self.Transposed_originalSM_matrix[SpotsStart[i][1]][SpotsStart[i][0]], SavGol_Area, SavGol_Polynom)
                else:
                    Label = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1])  
                    #Data = self.originalSM_matrix[SpotsStart[i][0]][SpotsStart[i][1]]
                    Data = self.Transposed_originalSM_matrix[SpotsStart[i][1]][SpotsStart[i][0]]
                XMaxWL = WLs[np.argmax(Data, axis=0)]
                XMax = max(Data)
                figSM_ax1.plot(WLs, Data, label=Label, color=ColorList[i])
                if ShowMax:
                    plt.axvline(x=XMaxWL, color=ColorList[i])
                    figSM_ax1.text(XMaxWL, XMax, str(XMaxWL)+" nm")
                i += 1

        figSM_ax1.set_xlabel('Wavelength [nm]')
        figSM_ax1.set_ylabel('Intensity [Counts]')

        figSM_ax1majx = plticker.MultipleLocator(base=20) # this locator puts ticks at regular intervals
        figSM_ax1.xaxis.set_major_locator(figSM_ax1majx)
        figSM_ax1minx = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
        figSM_ax1.xaxis.set_minor_locator(figSM_ax1minx)
        #figSM_ax1.set_ylim(-2,3)

        figSM_ax1.set_xlim(WLlow,WLhigh)
        figSM_ax1.set_ylim(Intlow,Inthigh)
        if Legend:
            figSM_ax1.legend()
            figSM_ax1.legend(fontsize=4)
        plt.show()

    def load_data(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                          "Load Overview-Files",
                                                          "", # Startverzeichnis
                                                          "Overview (*.overview);;All Files (*)",
                                                          options=options)
        if fileName:
            try:
                # Daten laden, Annahme: Tabulator als Trennzeichen
                self.z_data = np.loadtxt(fileName, delimiter='\t')

                if self.z_data.ndim != 2:
                    raise ValueError("File is not 2-dimensional.")

                # Gitterkoordinaten (X, Y) erstellen (basierend auf Indizes)
                ny, nx = self.z_data.shape
                x = np.arange(nx + 1) # +1 für pcolormesh Kanten
                y = np.arange(ny + 1) # +1 für pcolormesh Kanten
                self.X, self.Y = np.meshgrid(x, y)

                # Alten Plot löschen und neu zeichnen
                self.ax.clear()
                # cmap='viridis' ist eine gute Standard-Colormap
                self.mesh_plot = self.ax.pcolormesh(self.X, self.Y, self.z_data,
                                                    shading='flat', # oder 'auto'/'gouraud'
                                                    cmap='Greys',
                                                    snap=True) # Wichtig für genaue Klicks an Kanten

                #self.figure.colorbar(self.mesh_plot, ax=self.ax, orientation='vertical')
                #self.figure.colorbar.set_fontcolor("white")
                #self.figure.colorbar(self.mesh_plot, ax=self.ax)
                #self.figure.colorbar.set_label(label="Z-Wert", color="white")
                #self.ax.set_xlabel("X Index")
                #self.ax.set_ylabel("Y Index")
                self.ax.set_title("Click to select position!", color="white")
                self.ax.set_facecolor(((53/255),(53/255),(53/255)))
                self.figure.patch.set_facecolor(((53/255),(53/255),(53/255)))
                self.ax.xaxis.label.set_color('white')
                self.ax.yaxis.label.set_color('white')
                self.ax.tick_params(axis='x', colors='white')
                self.ax.tick_params(axis='y', colors='white')
                # Optional: Seitenverhältnis anpassen, falls die Zellen quadratisch sein sollen
                self.ax.set_aspect('equal', adjustable='box')
                self.ax.grid(True) # Gitter wieder hinzufügen
                self.canvas.draw() # Wichtig: Canvas neu zeichnen

                # Geklickte Punkte zurücksetzen, wenn neue Daten geladen werden
                self.clear_points() # Ruft auch update_points_display auf

            except FileNotFoundError:
                 QMessageBox.critical(self, "Error", f"File not found:\n{fileName}")
            except ValueError as ve:
                 QMessageBox.critical(self, "Error", f"Invalid data format in file:\n{fileName}\n\n{ve}")
            except Exception as e:
                 QMessageBox.critical(self, "Error", f"An unexpected Error occured:\n{e}")
                 self.z_data = None # Daten zurücksetzen bei Fehler
                 self.X = None
                 self.Y = None
                 self.ax.clear()
                 self.ax.set_title("Error loading file")
                 self.canvas.draw()


    def on_click3(self, event):
        # Prüfen, ob der Klick innerhalb der Achsen war und Daten geladen sind
        if event.inaxes == self.ax3 and self.z_data3 is not None:
            # event.xdata und event.ydata sind die Koordinaten im Datenraum
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                x = int(round(x))
                y = int(round(y))
                #print(f"Click registered with (X={x}, Y={y})")
                self.clicked_points3.append((x, y))
                self.update_points_display3()
        else:
             print("Click outside the axes or no data loaded.")

    def update_points_display3(self):
        # Formatieren der Punkte für eine schöne Ausgabe
        display_text = "\n".join([f"({x:.3f}, {y:.3f})" for x, y in self.clicked_points3])
        self.points_display3.setText(display_text)

    def clear_points3(self):
        self.clicked_points3 = []
        #print("Saved positions deleted.")
        self.update_points_display3() # Textfeld leeren

    def on_click(self, event):
        # Prüfen, ob der Klick innerhalb der Achsen war und Daten geladen sind
        if event.inaxes == self.ax and self.z_data is not None:
            # event.xdata und event.ydata sind die Koordinaten im Datenraum
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                x = int(round(x))
                y = int(round(y))
                #print(f"Click registered with (X={x}, Y={y})")
                self.clicked_points.append((x, y))
                self.update_points_display()
        else:
             print("Click outside the axes or no data loaded.")


    def update_points_display(self):
        # Formatieren der Punkte für eine schöne Ausgabe
        display_text = "\n".join([f"({x:.3f}, {y:.3f})" for x, y in self.clicked_points])
        self.points_display.setText(display_text)


    def clear_points(self):
        self.clicked_points = []
        #print("Saved positions deleted.")
        self.update_points_display() # Textfeld leeren

    def SomeChanges(self):
        try:
            if self.thread.isRunning():
                self.thread.lowerLimit = self.lower_spinbox.value()
                self.thread.upperLimit = self.upper_spinbox.value()
            else:
                pass
        except:
            pass

    def MittelChanges(self):
        try:
            if self.thread.isRunning():
                self.thread.x_mittel_center = self.x_mittel_spinbox.value()
                self.thread.y_mittel_center = self.y_mittel_spinbox.value()
                self.thread.size_mittel = self.size_spinbox.value()
            else:
                pass
        except:
            pass

    def BereichChanges(self):
        global Spectra
        global AbsMin
        global AbsMax
        global Fixed2d
        global Fixed3d
        global Fixed3dInit
        global Cb3d
        global SaveSVG
        global InitMin
        global InitMax
        global PixelPics

        Spectra = int(self.Range_spinbox.value())
        AbsMin = self.lower_spinbox.value()
        AbsMax = self.upper_spinbox.value()
        Fixed2d = self.limits2d_check.isChecked()
        Fixed3d = self.limits3d_check.isChecked()
        Fixed3dInit = self.limit3dInit_check.isChecked()
        Cb3d = self.colorbar_check.isChecked()
        PixelPics = self.PixelPics_check.isChecked()
        SaveSVG = self.SVG_check.isChecked()
        InitMax = self.upper_spinbox_Init.value()
        InitMin = self.lower_spinbox_Init.value()
        
        try:
            if self.thread.isRunning():
                self.thread.x_center = self.x_spinbox.value()
                self.thread.y_center = self.y_spinbox.value()
                self.thread.bereich_mittelwert_range = self.Range_spinbox.value()
            else:
                pass
        except:
            pass

    def ordner_auswaehlen_spectra(self):
        ordnerpfad_spectra = QFileDialog.getExistingDirectory(self, "Select path")
        if ordnerpfad_spectra:
            self.ordner_label2.setText(f"Path: {ordnerpfad_spectra}")

    def ordner_auswaehlen(self):
        ordnerpfad = QFileDialog.getExistingDirectory(self, "Select path")
        ordnerpfadSpectra = ordnerpfad + "/Spectra"
        if ordnerpfad:
            self.ordner_label.setText(f"Path: {ordnerpfad}")
            self.ordner_label2.setText(f"Path: {ordnerpfadSpectra}")

    def plots_generieren_spectralMap(self):
        self.plot_button3.setStyleSheet("color: white; background-color: rgb(255,0,0); border-radius: 10px")
        self.plot_button3.setEnabled(False)
        
        self.Spectra_from_SpectralMap()
        
        self.plot_button3.setEnabled(True)
        self.plot_button3.setStyleSheet("color: black; background-color: rgb(0,255,0); border-radius: 10px")

    def plots_generieren_spectra(self):
        ordnerpfad_spectra = self.ordner_label2.text().split(" ")[-1]
        if not os.path.isdir(ordnerpfad_spectra):
            QMessageBox.warning(self, "Error", "No valid path selected!")
            return
        self.plot_button2.setStyleSheet("color: white; background-color: rgb(255,0,0); border-radius: 10px")
        self.plot_button2.setEnabled(False)

        self.Spectra_from_Pixel(ordnerpfad_spectra,self.clicked_points)
        #self.Spectra_from_Pixel(ordnerpfad_spectra,self.clicked_points,RangeInput)

        #QMessageBox.information(self, "Ready", "Plots succesfully generated!")
        self.plot_button2.setEnabled(True)
        self.plot_button2.setStyleSheet("color: black; background-color: rgb(0,255,0); border-radius: 10px")
        #self.ordner_label2.setText("No path selected")


    def Spectra_from_Pixel(self, ordnerpfad_spectra, SpotsStart):
        #print(ordnerpfad_spectra)
        NumberOfSamples = len(SpotsStart)

        ODhigh = self.upper_spinbox2.value()
        ODlow = self.lower_spinbox2.value()
        Legend = self.colorbar_check2.isChecked()
        WLhigh = self.WL_spinbox_high.value()
        WLlow = self.WL_spinbox_low.value()
        RangeInputX = self.x_mittel_spinbox2.value()
        RangeInputY = self.y_mittel_spinbox2.value()
        SavGol_Area = self.SGArea_spinbox2.value()
        SavGol_Polynom = self.SGPoly_spinbox2.value()
        SavGol = self.groupboxSavGol2.isChecked()
        PlotBoth = self.UseBoth_check2.isChecked()
        ShowMax = self.ShowMax_check2.isChecked()
        ShowMin = self.ShowMin_check2.isChecked()
        #ShowMin = True
        #ShowMin = self.ShowMin_check2.isChecked()
        #ShowMaxNum = self.ShowMaxNum.value()
        

        # Load Data
        WLs = np.loadtxt(ordnerpfad_spectra + '/Pixel_0x0.txt',  delimiter='\t', usecols=(0), unpack=True)
        
        #Spot1
        RangeInputX = RangeInputX - 1
        RangeInputY = RangeInputY - 1
        #print("RangeInputX: " + str(RangeInputX) + " - RangeInputY: " + str(RangeInputY))
        i = 0
        Samples = []
        while i < NumberOfSamples:
                xStart = SpotsStart[i][0]
                yStart = SpotsStart[i][1]
                #print("xStart: " + str(xStart) + " - yStart: " + str(yStart))
                xStart = int(round(xStart-(RangeInputX/2)))
                yStart = int(round(yStart-(RangeInputY/2)))
                #print("xStart: " + str(xStart) + " - yStart: " + str(yStart))

                xStop = int(round(xStart+RangeInputX))
                yStop = int(round(yStart+RangeInputY))
                #print("xStop: " + str(xStop) + " - yStop: " + str(yStop))
                #print(xStart,xStop,yStart,yStop)

                x = xStart
                y = yStart
                Filename = ordnerpfad_spectra + "/Pixel_" + str(y) + "x" + str(x) + ".txt"
                #print(Filename)
                Sample = np.loadtxt(Filename,  delimiter='\t', usecols=(1), unpack=True)
                while y <= yStop:
                        x = xStart
                        while x <= xStop:
                                if y == yStart and x == xStart:
                                        pass
                                else:
                                        Filename = ordnerpfad_spectra + "/Pixel_" + str(y) + "x" + str(x) + ".txt"
                                        #print(Filename)
                                        try:
                                            Appending = np.loadtxt(Filename,  delimiter='\t', usecols=(1), unpack=True)
                                        except:
                                            print("Out of range!")
                                        Sample += Appending
                                x += 1
                        y += 1
                Sample = Sample/((RangeInputX+1)*(RangeInputY+1))
                Samples.append(Sample)
                i += 1

        # Create Figure, Grid and Axis
        fig1 = plt.figure(figsize=(8/2.54, 8/2.54), dpi=300, constrained_layout=True)
        gs = fig1.add_gridspec(1, 1)
        fig1_ax1 = fig1.add_subplot(gs[0, 0])



        #fig1.suptitle('WSe2 Flake on glass')


        ColorList = ['b','g','r','orange','c','m','y','k','lime','darorange','firebrick','hotpink','deepskyblue','crimson','pink','indigo','teal','limegreen','mediumvioletred','blueviolet','darkgoldenrod','lavender','navajowhite','saddlebrown','palevioletred','dodgerblue','lightslategrey']
        ColorListAlt = ['c','m','y','b','g','r','orange','k','lime','darorange','firebrick','hotpink','deepskyblue','crimson','pink','indigo','teal','limegreen','mediumvioletred','blueviolet','darkgoldenrod','lavender','navajowhite','saddlebrown','palevioletred','dodgerblue','lightslategrey']
        # Plotting and Axis Formatting fog1_ax1
        i = 0
        while i < len(Samples):
                #print(ColorList[i])
                if SavGol == True and PlotBoth == True:
                    Label = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1]) + " SavGol" 
                    Label2 = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1])  
                    Data = savgol_filter(Samples[i], SavGol_Area, SavGol_Polynom)
                    Data2 = Samples[i]
                    fig1_ax1.plot(WLs, Data2, label=Label2, color=ColorList[i], linestyle='dashed')
                elif SavGol:
                    Label = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1]) + " SavGol"
                    Data = savgol_filter(Samples[i], SavGol_Area, SavGol_Polynom)
                else:
                    Label = "Position " + str(SpotsStart[i][0]) + "X" + str(SpotsStart[i][1])
                    Data = Samples[i]
                fig1_ax1.plot(WLs,Data, label=Label, color=ColorList[i])

                #npa = np.asarray(Data, dtype=np.float32)
                npa = Data

                if ShowMax:
                    npamax = argrelextrema(npa, np.greater)
                    j = 0
                    while j < len(npamax[0]):
                        XMaxWL = WLs[npamax[0][j]]
                        XMax = Data[npamax[0][j]]
                        #print(XMaxWL)
                        #print(XMax)
                        plt.axvline(x=XMaxWL, color=ColorList[i])
                        fig1_ax1.text(XMaxWL, XMax, str(XMaxWL)+" nm")
                        j += 1
                if ShowMin:
                    npamin = argrelextrema(npa, np.less)
                    j = 0
                    while j < len(npamin[0]):
                        XMinWL = WLs[npamin[0][j]]
                        XMin = Data[npamin[0][j]]
                        #print(XMinWL)
                        #print(XMin)
                        plt.axvline(x=XMinWL, color=ColorList[i])
                        fig1_ax1.text(XMinWL, XMin, str(XMinWL)+" nm")
                        j += 1
                i += 1

        fig1_ax1.set_xlabel('Wavelength [nm]')
        fig1_ax1.set_ylabel('BG corr. Absorbance [OD]')

        fig1_ax1majx = plticker.MultipleLocator(base=20) # this locator puts ticks at regular intervals
        fig1_ax1.xaxis.set_major_locator(fig1_ax1majx)
        fig1_ax1minx = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
        fig1_ax1.xaxis.set_minor_locator(fig1_ax1minx)
        #fig1_ax1.set_ylim(-2,3)

        fig1_ax1.set_xlim(WLlow,WLhigh)
        fig1_ax1.set_ylim(ODlow,ODhigh)
        if Legend:
            fig1_ax1.legend()
            fig1_ax1.legend(fontsize=4)
        plt.show()

    def plots_generieren(self):
        ordnerpfad = self.ordner_label.text().split(" ")[-1]
        if not os.path.isdir(ordnerpfad):
            QMessageBox.warning(self, "Error", "No valid path selected!")
            return
        self.plot_button.setStyleSheet("color: white; background-color: rgb(255,0,0); border-radius: 10px")

        self.thread = WorkerThread(
            ordnerpfad, self.PlotRange.currentIndex(), self.x_Offset.value(), self.y_Offset.value(), self.x_spinbox.value(), self.y_spinbox.value(),
            self.x_mittel_spinbox.value(), self.y_mittel_spinbox.value(),
            self.size_spinbox.value(), self.lower_spinbox.value(), self.upper_spinbox.value(), self.Range_spinbox.value(), self.klick_position_event,
            self.mutex)  # klick_position_event und mutex übergeben
        self.thread.finished.connect(self.thread_finished)
        self.thread.progress.connect(self.update_progress)
        self.thread.progress2.connect(self.update_progress2)
        self.thread.progress3.connect(self.update_progress3)
        self.thread.plot_bereit.connect(self.plot_anzeigen)
        self.thread.start()
        self.plot_button.setEnabled(False)

    def plot_anzeigen(self, daten_array):
        global InitMin
        global InitMax
        global Fixed3dInit
        
        # Alten Plot entfernen, falls vorhanden
        if self.plot_layout1.count() > 0:
            item = self.plot_layout1.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            item2 = self.plot_layout1.takeAt(0)
            widget2 = item2.widget()
            if widget2 is not None:
                widget2.deleteLater()

        # Figure und Canvas erstellen
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        if Fixed3dInit:
            mesh = ax.pcolormesh(daten_array, cmap='Greys',vmin=InitMin, vmax=InitMax)
        else:
            mesh = ax.pcolormesh(daten_array, cmap='Greys')
        ax.set_title("Click on the " + str(self.PlotRange.currentText()) + " of the desired plot area", color="white")
        ax.set_facecolor(((53/255),(53/255),(53/255)))
        fig.patch.set_facecolor(((53/255),(53/255),(53/255)))
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        #ax.set_zlim(0, 2000)
        #self.upper_spinbox.setValue(int(np.nanmax(daten_array)))
        #self.lower_spinbox.setValue(int(np.nanmin(daten_array)))
        #self.upper_spinbox.setValue(int(daten_array.nanmax()))
        #self.lower_spinbox.setValue(int(daten_array.nanmin()))

        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(canvas, self)
        self.plot_layout1.addWidget(toolbar, 0, 0)
        self.plot_layout1.addWidget(canvas, 1, 0)

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:  # Überprüfen, ob Klick im Plotbereich war
                self.klick_position = (int(event.ydata), int(event.xdata))
                self.x_spinbox.setValue(self.klick_position[1])  # X-Spinbox aktualisieren
                self.y_spinbox.setValue(self.klick_position[0])  # Y-Spinbox aktualisieren
                self.klick_position_event.wakeAll()  # Event für die Klickposition setzen
                #print(self.klick_position[1], self.klick_position[0])

        canvas.mpl_connect('button_press_event', onclick)

    def update_progress(self, progress):
        self.ordner_label.setText(f"Plotrange selection: {progress}%")

    def update_progress2(self, progress2):
        self.ordner_label.setText(f"Pixel calculation: {progress2}%")

    def update_progress3(self, progress3):
        self.ordner_label.setText(f"Maxima calculation: {progress3}%")
        #if progress3 == 100:
        #    self.ordner_label.setText("No path selected")


    def thread_finished(self):
        QMessageBox.information(self, "Ready", "Plots succesfully generated!")
        self.plot_button.setEnabled(True)
        self.plot_button.setStyleSheet("color: black; background-color: rgb(0,255,0); border-radius: 10px")
        self.ordner_label.setText("No path selected")

    def initial_plot(self):
        global Pixel

        # Figure und Canvas erstellen
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        daten = np.zeros((Pixel, Pixel))  # Array mit Nullen
        ax.pcolormesh(daten, cmap='Greys_r')
        ax.set_title("Live-Plot", color="white")
        ax.set_facecolor(((53/255),(53/255),(53/255)))
        fig.patch.set_facecolor(((53/255),(53/255),(53/255)))
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        #ax.spines['bottom'].set_color('white')
        #ax.spines['top'].set_color('white') 
        #ax.spines['right'].set_color('white')
        #ax.spines['left'].set_color('white')
        #ax.title.label.set_color('white')

        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(canvas, self)
        self.plot_layout1.addWidget(toolbar, 0, 0)
        self.plot_layout1.addWidget(canvas, 1, 0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(StyleName)
    if StyleColor == "dark" and StyleName != "windowsvista":
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53,53,53))
        palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15,15,15))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53,53,53))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53,53,53))
        palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
                            
        #palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142,45,197).lighter())
        #palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0,250,0).lighter())
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0,0,255).lighter())
        palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        app.setPalette(palette)
    elif StyleColor == "dark" and StyleName == "windowsvista":
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53,53,53))
        palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.black)
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15,15,15))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53,53,53))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(153,153,153))
        palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.black)
        palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        #palette.setColor(QtGui.QPalette.TabWidget, QtCore.Qt.red)
                            
        #palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142,45,197).lighter())
        #palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0,250,0).lighter())
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0,0,255).lighter())
        palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        app.setPalette(palette)
        
    #w = Fenster()
    window = MyWindow()
    #app.statusBar().showMessage("Property of University Tübingen")  

    sys.exit(app.exec_())