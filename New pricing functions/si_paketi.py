"""
si_paketi.py — katalog rednih paketov slovenskih dobaviteljev + validacija
združljivosti paketa z gospodinjstvom.

VSE CENE SO BREZ DDV, v EUR/kWh oz. EUR/mesec.

KLJUČNO PRAVILO O DDV (popravek prejšnje verzije):
  "Odkup presežka proizvedene električne energije v napravi za samooskrbo
   ni predmet obdavčitve z DDV."
  — gen-i.si (vsi samooskrbni ceniki), bisol-energija.si
  Dobropis za oddajo se torej NE vključi v osnovo za DDV; odšteje se od
  računa Z DDV.

VIRI (preverjeno 22. 7. 2026):
 [G1] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/redni-cenik-elektricne-energije-za-gospodinjske-odjemalce/
 [G2] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/gen-i-fiksni-gospodinjski-odjemalci/
 [G3] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/aktivni-cenik-elektrike-za-dom/
 [G4] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/gen-i-dinamicni-gospodinjski-odjemalci/
 [G5] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/redni-cenik-za-prevzem-in-oddajo-elektricne-energije-pri-samooskrbi-gospodinjskih-odjemalcev/
 [G6] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/aktivni-cenik-samooskrbe-za-dom-in-skupnosti/
 [G7] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/fiksni-cenik-samooskrbe-za-dom-in-skupnosti/
 [G8] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/gen-i-dinamicni-samooskrba-za-gospodinjske-odjemalce/
 [G9] https://gen-i.si/dom/elektricna-energija/ceniki-in-akcije/redni-cenik-elektricne-energije-za-samooskrbo-gospodinjskih-odjemalcev/
 [B1] https://www.bisol-energija.si/fiksni-gospodinjstva
 [B2] https://www.bisol-energija.si/dinamicni-gospodinjstva
 [B3] https://www.bisol-energija.si/samooskrba/dinamicna
 [P1] https://www.petrol.si/binaries/content/assets/www/2025/dokumenti-in-obrazci/ee/go/cenik_go-odjem_marec-2025_f1.pdf
 [P2] https://www.petrol.si/binaries/content/assets/www/2026/dokumenti/ee/akcijski-cenik-elektricne-energije-za-gospodinjske-odjemalce-fiks-2026-9.-05.-2026.pdf
 [P3] https://www.petrol.si/za-dom/energenti/samooskrba
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TipCene(str, Enum):
    TARIFNI = "tarifni"          # VT/MT ali ET
    AKTIVNI = "aktivni"          # 4 tarife: sončna NS/VS, osnovna, konična
    DINAMICNI = "dinamicni"      # SIPX + pribitek


class TipOdkupa(str, Enum):
    NI = "ni"                    # dobavitelj presežkov ne odkupuje
    FIKSNI = "fiksni"            # ena cena EUR/kWh
    AKTIVNI = "aktivni"          # 4 tarife
    DINAMICNI = "dinamicni"      # SIPX − pribitek
    NET_METERING = "net_metering"  # letno netiranje, brez sprotnega dobropisa


class Shema(str, Enum):
    """Shema samooskrbe, v katero je merilno mesto vključeno."""
    BREZ = "brez"                    # ni naprave za samooskrbo
    NOVA = "nova"                    # soglasje po 1. 1. 2024, 15-min obračun
    NET_METERING = "net_metering"    # soglasje do 31. 12. 2023, letno netiranje


class NezdruzljivPaket(ValueError):
    """Paket ni združljiv s konfiguracijo gospodinjstva."""


@dataclass(frozen=True)
class Gospodinjstvo:
    """Konfiguracija odjemnega mesta — podlaga za validacijo paketa."""
    ime: str = "gospodinjstvo"
    dogovorjena_moc: Dict[int, float] = field(default_factory=dict)  # {blok: kW}
    ima_pv: bool = False
    shema_samooskrbe: Shema = Shema.BREZ
    skupnostna: bool = False          # član skupnostne samooskrbe / skupnosti OVE
    meritve_15min: bool = True
    eko_racun: bool = True            # elektronski račun -> EKO popust
    znacilni_primer: int = 2          # 1–10, za omrežnino deljene energije

    def __post_init__(self):
        if self.ima_pv and self.shema_samooskrbe is Shema.BREZ:
            raise ValueError(
                f"{self.ime}: ima_pv=True zahteva shema_samooskrbe NOVA ali NET_METERING."
            )
        if not self.ima_pv and self.shema_samooskrbe is not Shema.BREZ:
            raise ValueError(
                f"{self.ime}: shema_samooskrbe={self.shema_samooskrbe.value} "
                f"brez naprave za samooskrbo (ima_pv=False)."
            )
        if self.skupnostna and not self.ima_pv:
            raise ValueError(
                f"{self.ime}: skupnostna=True zahteva vključitev v samooskrbo."
            )


@dataclass(frozen=True)
class Paket:
    """Cenik dobavitelja za gospodinjske odjemalce."""
    id: str
    dobavitelj: str
    ime: str
    vir: str
    velja_od: dt.date

    tip_cene: TipCene
    tip_odkupa: TipOdkupa = TipOdkupa.NI

    # --- prevzem: tarifni ---
    vt: float = 0.0
    mt: float = 0.0
    et: float = 0.0
    # --- prevzem: aktivni (4 tarife) ---
    soncna_ns: float = 0.0
    soncna_vs: float = 0.0
    osnovna: float = 0.0
    konicna: float = 0.0
    # --- prevzem: dinamični ---
    pribitek_odjem: float = 0.0        # EUR/kWh, prišteje se SIPX
    cap_sipx: Optional[float] = None   # zgornja meja URNEGA SIPX (EUR/kWh)
    cap_mesecni: Optional[float] = None  # zgornja meja MESEČNE povprečne cene

    # --- oddaja (odkup presežkov) ---
    odkup_fiksni: float = 0.0
    odkup_soncna_ns: float = 0.0
    odkup_soncna_vs: float = 0.0
    odkup_osnovna: float = 0.0
    odkup_konicna: float = 0.0
    pribitek_oddaja: float = 0.0       # EUR/kWh, ODŠTEJE se od SIPX

    # --- fiksne postavke ---
    mesecno_nadomestilo: float = 0.0
    mesecno_nadomestilo_eko: Optional[float] = None
    dodatna_storitev: float = 0.0      # npr. BISOL DINAMIČNI+ 1,63 EUR/mes

    # --- pravila združljivosti ---
    zahteva_pv: bool = False           # samo za lastnike naprave za samooskrbo
    dovoljuje_pv: bool = False         # sme ga imeti gospodinjstvo s PV
    dovoljene_sheme: tuple = (Shema.BREZ,)
    dovoljuje_skupnostno: bool = False
    zahteva_15min: bool = False
    opombe: str = ""

    def nadomestilo(self, eko: bool) -> float:
        if eko and self.mesecno_nadomestilo_eko is not None:
            return self.mesecno_nadomestilo_eko + self.dodatna_storitev
        return self.mesecno_nadomestilo + self.dodatna_storitev


# ===========================================================================
# KATALOG
# ===========================================================================
PAKETI: Dict[str, Paket] = {}


def _reg(p: Paket) -> Paket:
    PAKETI[p.id] = p
    return p


# --------------------------------------------------------------- GEN-I: dobava
_reg(Paket(
    id="GENI_REDNI", dobavitelj="GEN-I", ime="Redni cenik za gospodinjske odjemalce",
    vir="[G1]", velja_od=dt.date(2025, 3, 1), tip_cene=TipCene.TARIFNI,
    vt=0.11990, mt=0.09790, et=0.10890,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    opombe="Ne velja za oskrbo skupnih delov večstanovanjskih stavb.",
))

_reg(Paket(
    id="GENI_FIKSNI", dobavitelj="GEN-I", ime="GEN-I Fiksni – Gospodinjski odjemalci",
    vir="[G2]", velja_od=dt.date(2025, 10, 23), tip_cene=TipCene.TARIFNI,
    vt=0.12490, mt=0.10290, et=0.11390,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    opombe="12-mesečna vezava.",
))

_reg(Paket(
    id="GENI_AKTIVNI", dobavitelj="GEN-I", ime="Aktivni cenik elektrike za dom",
    vir="[G3]", velja_od=dt.date(2026, 2, 6), tip_cene=TipCene.AKTIVNI,
    soncna_ns=0.03490, soncna_vs=0.08990, osnovna=0.11790, konicna=0.17990,
    et=0.10890,   # nadomestna cena ob pomanjkljivih 15-min podatkih = ET rednega
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_15min=True,
))

_reg(Paket(
    id="GENI_DINAMICNI", dobavitelj="GEN-I", ime="GEN-I Dinamični – Gospodinjski odjemalci",
    vir="[G4]", velja_od=dt.date(2024, 10, 2), tip_cene=TipCene.DINAMICNI,
    pribitek_odjem=0.01199, cap_sipx=0.22000,
    mesecno_nadomestilo=2.97, mesecno_nadomestilo_eko=1.97,
    opombe="Zamejitev 220 EUR/MWh velja na URNI SIPX. Navzdol neomejeno.",
))

# ----------------------------------------------------- GEN-I: samooskrba (nova)
_reg(Paket(
    id="GENI_SAMO_REDNI", dobavitelj="GEN-I",
    ime="Redni cenik za prevzem in oddajo pri samooskrbi",
    vir="[G5]", velja_od=dt.date(2025, 3, 1),
    tip_cene=TipCene.TARIFNI, tip_odkupa=TipOdkupa.FIKSNI,
    et=0.10290, odkup_fiksni=0.05390,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    opombe="Pristop od 5. 2. 2026 le v sklopu paketa Pametna samooskrba.",
))

_reg(Paket(
    id="GENI_SAMO_AKTIVNI", dobavitelj="GEN-I",
    ime="Aktivni cenik samooskrbe za dom in skupnosti",
    vir="[G6]", velja_od=dt.date(2026, 2, 5),
    tip_cene=TipCene.AKTIVNI, tip_odkupa=TipOdkupa.AKTIVNI,
    soncna_ns=0.04090, soncna_vs=0.11490, osnovna=0.12990, konicna=0.19290,
    et=0.14090,                      # nadomestna cena ob pomanjkljivih meritvah
    odkup_soncna_ns=0.00190, odkup_soncna_vs=0.06990,
    odkup_osnovna=0.07490, odkup_konicna=0.14990,
    odkup_fiksni=0.01490,            # nadomestna cena oddaje
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    zahteva_15min=True,
    opombe="Ne velja za stranke paketa Pametna samooskrba.",
))

_reg(Paket(
    id="GENI_SAMO_FIKSNI", dobavitelj="GEN-I",
    ime="Fiksni cenik samooskrbe za dom in skupnosti",
    vir="[G7]", velja_od=dt.date(2026, 2, 17),
    tip_cene=TipCene.AKTIVNI, tip_odkupa=TipOdkupa.AKTIVNI,
    soncna_ns=0.04590, soncna_vs=0.11990, osnovna=0.13490, konicna=0.19790,
    et=0.14090,
    odkup_soncna_ns=0.00190, odkup_soncna_vs=0.06490,
    odkup_osnovna=0.06990, odkup_konicna=0.14490,
    odkup_fiksni=0.01490,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    zahteva_15min=True, opombe="12-mesečna vezava.",
))

_reg(Paket(
    id="GENI_SAMO_DINAMICNI", dobavitelj="GEN-I",
    ime="GEN-I Dinamični – Samooskrba za gospodinjske odjemalce",
    vir="[G8]", velja_od=dt.date(2024, 10, 2),
    tip_cene=TipCene.DINAMICNI, tip_odkupa=TipOdkupa.DINAMICNI,
    pribitek_odjem=0.01199, pribitek_oddaja=0.01199,
    mesecno_nadomestilo=2.97, mesecno_nadomestilo_eko=1.97,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,),
    dovoljuje_skupnostno=False,      # cenik izrecno IZKLJUČUJE skupnostno samooskrbo
    zahteva_15min=True,
    opombe="Cenik izrecno ne velja za skupnostno samooskrbo.",
))

# --------------------------------------------- GEN-I: samooskrba (NET metering)
_reg(Paket(
    id="GENI_NETMETERING", dobavitelj="GEN-I",
    ime="Redni cenik električne energije za samooskrbo (NET metering)",
    vir="[G9]", velja_od=dt.date(2025, 3, 1),
    tip_cene=TipCene.TARIFNI, tip_odkupa=TipOdkupa.NET_METERING,
    et=0.12990,
    mesecno_nadomestilo=1.99, mesecno_nadomestilo_eko=0.99,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NET_METERING,), dovoljuje_skupnostno=True,
    opombe="Letno netiranje; presežek nad letno porabo se prenese brezplačno, "
           "prizna se ugodnost v višini enega mesečnega nadomestila na celo MWh.",
))

# ------------------------------------------------------------- BISOL: dobava
_reg(Paket(
    id="BISOL_FIKSNI", dobavitelj="BISOL Energija", ime="Paket FIKSNI",
    vir="[B1]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.TARIFNI,
    vt=0.12700, mt=0.10700, et=0.11700, mesecno_nadomestilo=1.63,
    opombe="Brez vezave.",
))

_reg(Paket(
    id="BISOL_FIKSNI_VEZAVA", dobavitelj="BISOL Energija", ime="Paket FIKSNI z vezavo",
    vir="[B1]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.TARIFNI,
    vt=0.11600, mt=0.09600, et=0.10500, mesecno_nadomestilo=1.63,
    opombe="12-mesečna vezava.",
))

_reg(Paket(
    id="BISOL_FIKSNI99", dobavitelj="BISOL Energija", ime="Paket FIKSNI99",
    vir="[B1]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.TARIFNI,
    vt=0.10900, mt=0.09400, et=0.09900, mesecno_nadomestilo=1.63,
    opombe="Akcijski cenik, vezava do konca 2026.",
))

_reg(Paket(
    id="BISOL_DINAMICNI", dobavitelj="BISOL Energija", ime="Paket DINAMIČNI",
    vir="[B2]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.DINAMICNI,
    pribitek_odjem=0.01300, mesecno_nadomestilo=1.63,
    opombe="Brez vezave; navzdol neomejeno (tudi negativne cene).",
))

_reg(Paket(
    id="BISOL_DINAMICNI_PLUS", dobavitelj="BISOL Energija",
    ime="Paket DINAMIČNI + storitev DINAMIČNI+",
    vir="[B2]", velja_od=dt.date(2026, 1, 1), tip_cene=TipCene.DINAMICNI,
    pribitek_odjem=0.01300, cap_mesecni=0.14700,
    mesecno_nadomestilo=1.63, dodatna_storitev=1.63,
    opombe="Zamejitev velja na MESEČNO povprečno obračunsko ceno SIPX "
           "(0,147 EUR/kWh brez pribitka), ne na urno ceno. 12-mesečna vezava.",
))

# --------------------------------------------------------- BISOL: samooskrba
_reg(Paket(
    id="BISOL_SAMO_DINAMICNA", dobavitelj="BISOL Energija", ime="DINAMIČNA Samooskrba",
    vir="[B3]", velja_od=dt.date(2026, 1, 1),
    tip_cene=TipCene.DINAMICNI, tip_odkupa=TipOdkupa.DINAMICNI,
    pribitek_odjem=0.01300, pribitek_oddaja=0.01300,
    mesecno_nadomestilo=1.63,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA,), dovoljuje_skupnostno=True,
    zahteva_15min=True,
    opombe="Simetričen razmik ±0,013 EUR/kWh okoli SIPX. Za soglasja po 1. 1. 2024. "
           "Za net metering le ob nadgradnji s hranilnikom.",
))

# ------------------------------------------------------------------- PETROL
_reg(Paket(
    id="PETROL_REDNI", dobavitelj="Petrol", ime="Redni cenik za gospodinjske odjemalce",
    vir="[P1]", velja_od=dt.date(2025, 3, 1), tip_cene=TipCene.TARIFNI,
    vt=0.12795, mt=0.10795, et=0.11795, mesecno_nadomestilo=1.98,
))

_reg(Paket(
    id="PETROL_AKCIJSKI", dobavitelj="Petrol", ime="Akcijski cenik (marec 2025)",
    vir="[P1]", velja_od=dt.date(2025, 3, 1), tip_cene=TipCene.TARIFNI,
    vt=0.11995, mt=0.09995, et=0.10995, mesecno_nadomestilo=1.98,
))

_reg(Paket(
    id="PETROL_FIKS2026", dobavitelj="Petrol", ime="Akcijski cenik FIKS 2026",
    vir="[P2]", velja_od=dt.date(2026, 5, 9), tip_cene=TipCene.TARIFNI,
    vt=0.12395, mt=0.10195, et=0.11295, mesecno_nadomestilo=1.98,
    opombe="12 mesecev zagotovljene cene.",
))

_reg(Paket(
    id="PETROL_SAMOOSKRBA", dobavitelj="Petrol", ime="Samooskrba (brez odkupa presežkov)",
    vir="[P3]", velja_od=dt.date(2025, 3, 1),
    tip_cene=TipCene.TARIFNI, tip_odkupa=TipOdkupa.NI,
    vt=0.12795, mt=0.10795, et=0.11795, mesecno_nadomestilo=1.98,
    zahteva_pv=True, dovoljuje_pv=True,
    dovoljene_sheme=(Shema.NOVA, Shema.NET_METERING), dovoljuje_skupnostno=True,
    opombe="Petrol viškov NE odkupuje in ne omogoča prenosa — oddana energija "
           "je za odjemalca brez vrednosti.",
))


# ===========================================================================
# VALIDACIJA
# ===========================================================================
def preveri_paket(paket: Paket, g: Gospodinjstvo,
                  datum: Optional[dt.date] = None,
                  strogo: bool = True) -> List[str]:
    """
    Preveri združljivost paketa z gospodinjstvom.
    Vrne seznam opozoril; ob strogo=True ob nezdružljivosti sproži NezdruzljivPaket.
    """
    napake: List[str] = []
    opozorila: List[str] = []

    # 1) PV vs. paket brez samooskrbe
    if g.ima_pv and not paket.dovoljuje_pv:
        napake.append(
            f"{g.ime} ima napravo za samooskrbo, paket '{paket.ime}' "
            f"({paket.dobavitelj}) pa je cenik za dobavo brez samooskrbe. "
            f"Uporabi samooskrbni cenik istega dobavitelja."
        )

    # 2) samooskrbni paket brez PV
    if paket.zahteva_pv and not g.ima_pv:
        napake.append(
            f"Paket '{paket.ime}' ({paket.dobavitelj}) je namenjen izključno "
            f"odjemalcem z napravo za samooskrbo, {g.ime} pa je nima."
        )

    # 3) shema samooskrbe (nova vs. NET metering)
    if g.ima_pv and g.shema_samooskrbe not in paket.dovoljene_sheme:
        dovoljene = ", ".join(s.value for s in paket.dovoljene_sheme)
        napake.append(
            f"{g.ime} je v shemi '{g.shema_samooskrbe.value}', paket "
            f"'{paket.ime}' pa velja za: {dovoljene}."
        )

    # 4) skupnostna samooskrba
    if g.skupnostna and not paket.dovoljuje_skupnostno:
        napake.append(
            f"{g.ime} je v skupnostni samooskrbi, paket '{paket.ime}' "
            f"pa je izrecno omejen na individualno samooskrbo."
        )

    # 5) 15-minutne meritve
    if paket.zahteva_15min and not g.meritve_15min:
        opozorila.append(
            f"Paket '{paket.ime}' predpostavlja 15-min meritve; brez njih se "
            f"uporabi nadomestna cena (ET)."
        )

    # 6) veljavnost cenika
    if datum is not None and datum < paket.velja_od:
        opozorila.append(
            f"Paket '{paket.ime}' velja šele od {paket.velja_od.isoformat()}, "
            f"obračunavaš pa {datum.isoformat()} — cene so ekstrapolirane nazaj."
        )

    # 7) PV brez odkupa
    if g.ima_pv and paket.tip_odkupa is TipOdkupa.NI:
        opozorila.append(
            f"Paket '{paket.ime}' ne odkupuje presežkov — oddana energija se "
            f"ovrednoti z 0 EUR."
        )

    if napake and strogo:
        raise NezdruzljivPaket(" | ".join(napake))
    return napake + opozorila


def zdruzljivi_paketi(g: Gospodinjstvo,
                      datum: Optional[dt.date] = None) -> List[Paket]:
    """Vsi paketi iz kataloga, ki so združljivi z danim gospodinjstvom."""
    out = []
    for p in PAKETI.values():
        try:
            preveri_paket(p, g, datum, strogo=True)
            out.append(p)
        except NezdruzljivPaket:
            pass
    return out
