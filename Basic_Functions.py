### Basic_Functions.py - Updated to accept parameters as function arguments

import pandas as pd
import numpy as np
from enum import Enum

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
    

def _nagrada_1(self, s):
    alfa = 0.1
    beta = 0.8
    soc = s.Baterija_norm
    if soc < alfa:
        return -5.0 * (alfa - soc) * self.faktor_n1
    if soc > beta:
        return -5.0 * (soc - beta) * self.faktor_n1
    return 0.0

def _nagrada_2(self, s, sprememba_baterije, _cena_el_med):
    if self.bat_kapaciteta <= 0:
        return 0.0
    gamma2 = 3.0
    norm_change = sprememba_baterije / self.bat_kapaciteta
    return -5.0 * gamma2 * norm_change * s.CenaElRel * self.faktor_n2

def _nagrada_3(self, placilo_zdaj):
    delta = 5.0 / 8.0
    return -delta * placilo_zdaj * self.faktor_n3

def _nagrada_skupno(self, s, sprememba_baterije, placilo_zdaj, cena_el_med):
    if self.bat_kapaciteta <= 0:
        return self._nagrada_3(placilo_zdaj)
    denominator = self.faktor_n1 + self.faktor_n2 + self.faktor_n3
    if denominator <= 0:
        return self._nagrada_3(placilo_zdaj)
    return (
        self._nagrada_1(s)
        + self._nagrada_2(s, sprememba_baterije, cena_el_med)
        + self._nagrada_3(placilo_zdaj)
    ) / denominator

    
import datetime

def calculate_interval_price(
    smp_market_price_mwh: float,
    total_consumed_kwh: float,
    utc_date: datetime.datetime,
    interval_minutes: int = 30
) -> dict:
    """
    Calculates the dynamic electricity price for a specific timestep in Australia.
    
    Args:
        smp_market_price_mwh (float): AEMO Wholesale Spot Price in AUD/MWh.
        total_consumed_kwh (float): Electricity consumed in kWh. Negative if exporting solar.
        utc_date (datetime.datetime): The UTC timestamp of the interval.
        interval_minutes (int): Length of the timestep (e.g., 15 or 30 minutes).
        
    Returns:
        dict: A breakdown of the constant and variable prices for that specific interval in AUD.
    """
    
    # -------------------------------------------------------------------------
    # 1. CONSTANTS & TAXES
    # -------------------------------------------------------------------------
    GST_RATE = 0.10
    DAYS_IN_MONTH = 30
    
    # -------------------------------------------------------------------------
    # 2. CONSTANT COSTS (Fixed Monthly/Daily Fees)
    # -------------------------------------------------------------------------
    # a. Retailer Subscription (Source: Amber Electric - roughly $22/mo inc. GST)
    monthly_subscription_ex_gst = 20.00 
    
    # b. Network Daily Supply Charge (Source: Typical Ausgrid/Endeavour tariff - ~$1.20/day inc GST)
    daily_supply_ex_gst = 1.09 
    
    # Prorate fixed costs down to this specific interval
    intervals_per_day = (24 * 60) / interval_minutes
    intervals_per_month = intervals_per_day * DAYS_IN_MONTH
    
    constant_cost_ex_gst = (daily_supply_ex_gst / intervals_per_day) + (monthly_subscription_ex_gst / intervals_per_month)
    constant_cost_inc_gst = constant_cost_ex_gst * (1 + GST_RATE)
    
    # -------------------------------------------------------------------------
    # 3. VARIABLE COSTS (Price of Electricity)
    # -------------------------------------------------------------------------
    # Convert spot price from MWh to kWh
    spot_price_kwh = smp_market_price_mwh / 1000.0
    
    # AEMO Loss Factors: Accounts for energy lost as heat during transmission
    # Source: AEMO Published Loss Factors 2025/2026
    MLF = 0.995  # Marginal Loss Factor (Transmission)
    DLF = 1.045  # Distribution Loss Factor (Local Grid)
    adjusted_spot_kwh = spot_price_kwh * MLF * DLF
    
    # Convert UTC to AEMO Time (AEST is strictly UTC+10 year-round)
    nem_time = utc_date + datetime.timedelta(hours=10)
    hour = nem_time.hour
    
    # Time-of-Use Network Tariff Rates (ex GST)
    # Source: Ausgrid Residential EA025 Time of Use structure
    if 15 <= hour < 21:
        network_rate_kwh = 0.2360  # Peak (3 PM - 9 PM)
    elif 10 <= hour < 15:
        network_rate_kwh = 0.0270  # Solar Sponge (10 AM - 3 PM)
    else:
        network_rate_kwh = 0.0720  # Off-Peak (All other times)
        
    # Environmental Schemes & Market Administration Fees (ex GST)
    # Source: Clean Energy Regulator (LGCs/STCs) & AEMO Fees
    env_market_rate_kwh = 0.0250
    
    variable_cost_ex_gst = 0.0
    variable_cost_inc_gst = 0.0
    
    # Determine Cost based on Import vs. Export
    if total_consumed_kwh >= 0:
        # IMPORTING: Paying for spot price, grid usage, and environmental fees
        total_rate_kwh_ex_gst = adjusted_spot_kwh + network_rate_kwh + env_market_rate_kwh
        variable_cost_ex_gst = total_consumed_kwh * total_rate_kwh_ex_gst
        
        # Apply 10% GST to imported electricity
        variable_cost_inc_gst = variable_cost_ex_gst * (1 + GST_RATE)
    else:
        # EXPORTING (Solar PV): The consumer is paid the spot price.
        # Volumetric network and environmental charges do not apply. 
        # Note: If AEMO spot price is negative, total_consumed_kwh (negative) * spot (negative) 
        # results in a positive cost, meaning the user pays to export.
        variable_cost_ex_gst = total_consumed_kwh * adjusted_spot_kwh
        
        # GST is typically not applicable to residential solar feed-in credits
        variable_cost_inc_gst = variable_cost_ex_gst 
        
    # -------------------------------------------------------------------------
    # 4. OUTPUT
    # -------------------------------------------------------------------------
    return {
        "constant_price_aud": round(constant_cost_inc_gst, 10),
        "variable_price_aud": round(variable_cost_inc_gst, 10),
        #"total_interval_price_aud": round(constant_cost_inc_gst + variable_cost_inc_gst, 5)
    }

