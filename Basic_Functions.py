### Basic_Functions.py - Updated to accept parameters as function arguments

import pandas as pd
import numpy as np
from enum import Enum

try:
    import auto_diagnostics

    auto_diagnostics.enable_auto_diagnostics()
except Exception:
    # Diagnostics are best-effort and must never break simulation code.
    pass

### Helper Functions
def BatMaxPraTrenutno(s, BatUcinkovitost, BatMaxPraznjenje):
    """Vrne maksimalno energijo ki jo lahko baterija odda v tem koraku"""
    return BatUcinkovitost * min(s.Baterija, BatMaxPraznjenje)

def BatMaxPolTrenutno(s, BatUcinkovitost, BatMaxPolnjenje, BatKapaciteta):
    """Vrne maksimalno energijo ki jo lahko baterija sprejme v tem koraku"""
    return (1/BatUcinkovitost) * min(BatKapaciteta - s.Baterija, BatMaxPolnjenje)

def PaneliOdvec(PaneliProizvodnja, Poraba):
    """Energija ki jo lahko paneli oddajo bateriji ali omrežju"""
    if PaneliProizvodnja > Poraba:
        return PaneliProizvodnja - Poraba
    else:
        return 0

def BaterijaSprememba(PaneliBaterija, OmrezjeBaterija, BaterijaDom, BaterijaOmrezje, BatUcinkovitost):
    """Sprememba baterije v tem koraku"""
    return BatUcinkovitost * (PaneliBaterija + OmrezjeBaterija) - (1/BatUcinkovitost) * (BaterijaDom + BaterijaOmrezje)

### State and Action Classes
class State(object):
    """Stanja našega agenta"""
    def __init__(self, CenaEl=0, CenaElMed=0, CenaElRel=1, Baterija=0, BaterijaProp=0.5, Generiranje=0,
                 Poraba=0, Placilo=0, Korak=0):
        self.CenaEl = CenaEl            # Cena elektrike eur/kWh
        self.CenaElMed = CenaElMed      # Povprečna cena elektrike npr v zadnjem mesecu cena elektrike eur/kWh
        self.CenaElRel = CenaElRel      # Relativna cena elektrike = CenaEl/CenaElMed
        self.Baterija = Baterija        # Stanje baterije v kWh
        self.BaterijaProp = BaterijaProp# Proporcionalno stanje baterije v %
        self.Generiranje = Generiranje  # Količina generirane elektrike kWh
        self.Poraba = Poraba            # Količina porabljene elektrike kWh
        self.Placilo = Placilo          # Plačilo za porabljeno elektriko
        self.Korak = Korak              # Trenutni korak

    def to_array(self):
        return np.array([self.CenaEl, self.CenaElMed, self.CenaElRel, self.Baterija, self.BaterijaProp,
                         self.Generiranje, self.Poraba, self.Placilo, self.Korak])

class Action(Enum):
    """Akcije našega agenta"""
    KUPI_POLNI = 0      # Kupi energijo da napajaš hišo in baterijo
    KUPI_HISA = 1       # Kupi energijo da napajaš hišo
    BAT_HISA = 2        # Napajaš hišo iz baterije
    BAT_PRODAJ = 3      # Napajaš hišo in prodajaš iz baterije
    # BAT_POCIVAJ = 4     #Ne uporabi baterije
    
    
    
