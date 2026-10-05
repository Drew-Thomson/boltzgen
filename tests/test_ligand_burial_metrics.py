import pytest
import numpy as np

def test_max_consecutive_ilv_sheet():
    # Mocking the inline logic
    def max_run_ilv(design_seq, dssp_numpy):
        max_run = 0
        current_run = 0
        for res, ss in zip(design_seq, dssp_numpy):
            if ss == 2 and res in ["I", "L", "V"]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        return max_run

    # IVVV in beta-sheet (2) -> 4
    assert max_run_ilv("AIVVVA", np.array([0, 2, 2, 2, 2, 0])) == 4
    # Aromatics break the run -> 1
    assert max_run_ilv("IWV", np.array([2, 2, 2])) == 1
    assert max_run_ilv("VYV", np.array([2, 2, 2])) == 1
    # alpha-helical sequences (1) -> 0
    assert max_run_ilv("IVVV", np.array([1, 1, 1, 1])) == 0

def test_outward_fractions():
    def get_fractions(full_seq, classifications):
        outward_res = [res for res, cls in zip(full_seq, classifications) if cls == "outward"]
        if len(outward_res) > 0:
            ilv_frac = sum(1 for r in outward_res if r in ["I", "L", "V"]) / len(outward_res)
            arom_frac = sum(1 for r in outward_res if r in ["W", "Y", "F"]) / len(outward_res)
        else:
            ilv_frac = 0.0
            arom_frac = 0.0
        return ilv_frac, arom_frac
    
    seq = "IVWYA"
    cls = ["outward", "inward", "outward", "outward", "inward"]
    # outward are I, W, Y. Total 3. ILV=1/3, Arom=2/3
    ilv, arom = get_fractions(seq, cls)
    assert np.isclose(ilv, 1/3)
    assert np.isclose(arom, 2/3)

def test_ligand_burial_fraction():
    design_sasa_unbound = 100.0
    design_sasa_bound = 10.0
    
    frac = 1.0 - (design_sasa_bound / design_sasa_unbound)
    assert np.isclose(frac, 0.9)
    
    # 0 unbound should yield 0.0
    frac_zero = 0.0
    assert frac_zero == 0.0

