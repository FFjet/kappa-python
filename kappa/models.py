"""Enum definitions mirroring kappa::models_* types."""
from __future__ import annotations

from enum import Enum


class ModelsProbVV(Enum):
    FHO = "model_prob_vv_fho"


class ModelsProbVT(Enum):
    FHO = "model_prob_vt_fho"


class ModelsProbDiss(Enum):
    THRESH_CMASS_VIBR = "model_prob_diss_thresh_cmass_vibr"
    THRESH_VIBR = "model_prob_diss_thresh_vibr"
    THRESH_CMASS = "model_prob_diss_thresh_cmass"
    THRESH = "model_prob_diss_thresh"


class ModelsCsElastic(Enum):
    RS = "model_cs_el_rs"
    VSS = "model_cs_el_vss"


class ModelsCsVV(Enum):
    RS_FHO = "model_cs_vv_rs_fho"
    VSS_FHO = "model_cs_vv_vss_fho"


class ModelsCsVT(Enum):
    RS_FHO = "model_cs_vt_rs_fho"
    VSS_FHO = "model_cs_vt_vss_fho"


class ModelsCsDiss(Enum):
    RS_THRESH_CMASS_VIBR = "model_cs_diss_rs_thresh_cmass_vibr"
    RS_THRESH_VIBR = "model_cs_diss_rs_thresh_vibr"
    RS_THRESH_CMASS = "model_cs_diss_rs_thresh_cmass"
    RS_THRESH = "model_cs_diss_rs_thresh"
    VSS_THRESH_CMASS_VIBR = "model_cs_diss_vss_thresh_cmass_vibr"
    VSS_THRESH_VIBR = "model_cs_diss_vss_thresh_vibr"
    VSS_THRESH_CMASS = "model_cs_diss_vss_thresh_cmass"
    VSS_THRESH = "model_cs_diss_vss_thresh"
    ILT = "model_cs_diss_ilt"


class ModelsKVV(Enum):
    RS_FHO = "model_k_vv_rs_fho"
    VSS_FHO = "model_k_vv_vss_fho"
    SSH = "model_k_vv_ssh"
    BILLING = "model_k_vv_billing"


class ModelsKVT(Enum):
    RS_FHO = "model_k_vt_rs_fho"
    VSS_FHO = "model_k_vt_vss_fho"
    SSH = "model_k_vt_ssh"
    PHYS4ENTRY = "model_k_vt_phys4entry"
    BILLING = "model_k_vt_billing"


class ModelsKExch(Enum):
    ARRH_SCANLON = "model_k_exch_arrh_scanlon"
    ARRH_PARK = "model_k_exch_arrh_park"
    WARNATZ = "model_k_exch_warnatz"
    RF = "model_k_exch_rf"
    POLAK = "model_k_exch_polak"
    MALIAT_D6K_ARRH_SCANLON = "model_k_exch_maliat_D6k_arrh_scanlon"
    MALIAT_3T_ARRH_SCANLON = "model_k_exch_maliat_3T_arrh_scanlon"
    MALIAT_INF_ARRH_SCANLON = "model_k_exch_maliat_infty_arrh_scanlon"
    MALIAT_D6K_ARRH_PARK = "model_k_exch_maliat_D6k_arrh_park"
    MALIAT_3T_ARRH_PARK = "model_k_exch_maliat_3T_arrh_park"
    MALIAT_INF_ARRH_PARK = "model_k_exch_maliat_infty_arrh_park"
    # Legacy compatibility: treat BORNMAYER entry as placeholder for exchange models mapping


class ModelsKDiss(Enum):
    RS_THRESH_CMASS_VIBR = "model_k_diss_rs_thresh_cmass_vibr"
    RS_THRESH_VIBR = "model_k_diss_rs_thresh_vibr"
    RS_THRESH_CMASS = "model_k_diss_rs_thresh_cmass"
    RS_THRESH = "model_k_diss_rs_thresh"
    VSS_THRESH_CMASS_VIBR = "model_k_diss_vss_thresh_cmass_vibr"
    VSS_THRESH_VIBR = "model_k_diss_vss_thresh_vibr"
    VSS_THRESH_CMASS = "model_k_diss_vss_thresh_cmass"
    VSS_THRESH = "model_k_diss_vss_thresh"
    ARRH_SCANLON = "model_k_diss_arrh_scanlon"
    ARRH_PARK = "model_k_diss_arrh_park"
    TM_D6K_ARRH_SCANLON = "model_k_diss_tm_D6k_arrh_scanlon"
    TM_3T_ARRH_SCANLON = "model_k_diss_tm_3T_arrh_scanlon"
    TM_INF_ARRH_SCANLON = "model_k_diss_tm_infty_arrh_scanlon"
    TM_D6K_ARRH_PARK = "model_k_diss_tm_D6k_arrh_park"
    TM_3T_ARRH_PARK = "model_k_diss_tm_3T_arrh_park"
    TM_INF_ARRH_PARK = "model_k_diss_tm_infty_arrh_park"
    # Legacy alias (C++ naming uses "INFTY")
    TM_INFTY_ARRH_SCANLON = "model_k_diss_tm_infty_arrh_scanlon"
    TM_INFTY_ARRH_PARK = "model_k_diss_tm_infty_arrh_park"
    PHYS4ENTRY = "model_k_diss_phys4entry"
    ILT = "model_k_diss_ilt"


class ModelsOmega(Enum):
    RS = "model_omega_rs"
    VSS = "model_omega_vss"
    BORNMAYER = "model_omega_bornmayer"
    LENNARD_JONES = "model_omega_lennardjones"
    ESA = "model_omega_esa"
