"""
Flexible accuracy metrics for chemical reaction completion benchmarks.

Exposed reaction matching views (from strict to more permissive):

  exact_match
      Pure canonical SMILES comparison (Counter-based, stoichiometry respected).
      No EQUIV_MAP expansion, no standardization — only atom-map removal and
      RDKit canonicalization.  Strictest level.

  proton_shuffle_match
      Proton-transfer-aware normalization is evaluated in two ways:
      (1) before EQUIV_MAP expansion and (2) after EQUIV_MAP expansion.
      A pair is considered a proton-shuffle match if either pathway matches.
      Standalone H⁺ is treated as mobile, monatomic halides are preserved as
      spectators, and other species are reduced via ChargeParent.

  pragmatic_equivalence_match
      Practical catch-all: accepts a pair if either (a) all-normalization
      overlap holds, (b) difference-normalized matching holds, or (c) the
      spectator-cancelled reaction core matches under a strict same-path
      normalization view.

  deep_pragmatic_equivalence_match
      Extends pragmatic_equivalence_match with extra targeted rewrites for
      frequent representation artifacts (H+/H-/H2 bookkeeping, H+ + OH-,
      H+ + halide acids, SOCl2-family shorthand, borohydride shorthand).

The post-EQUIV proton-shuffle pathway normalizes NaOH ([OH][Na]) →
Na⁺ + OH⁻, KOH → K⁺ + OH⁻, LiOH → Li⁺ + OH⁻, HCl → H⁺ + Cl⁻,
HBr → H⁺ + Br⁻, HI → H⁺ + I⁻, H₃O⁺ → H⁺ + H₂O, and H₂CO₃ → CO₂ + H₂O
before comparison.

Typical usage
-------------
    from utils.metrics import compare_reactions, flexible_accuracy, top_k_accuracy

    result = compare_reactions(pred, target)
    result.exact_match            # bool
    result.proton_shuffle_match   # bool
    result.all_normalizations_match  # bool
    result.pragmatic_equivalence_match  # bool
    result.deep_pragmatic_equivalence_match  # bool

    # aggregate accuracy at the proton-shuffle level
    acc = flexible_accuracy(predictions, targets,
                            match_fn=lambda p, t: compare_reactions(p, t).proton_shuffle_match)

    # top-k beam accuracy (CRB)
    acc5 = top_k_accuracy(beam_preds, targets, k=5)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


# ---------------------------------------------------------------------------
# Molecule-level helpers
# ---------------------------------------------------------------------------

def _sanitize_problematic_smiles(smiles: str) -> str:
    """Normalize a few known non-canonical notations before RDKit parsing."""
    s = smiles.strip()

    # Dative-bond AlCl3 forms frequently appear as [Cl][Al]<-[Cl]Cl.
    # Convert directional dative bonds to plain single bonds so RDKit can parse.
    s = s.replace("<-", "-").replace("->", "-")

    return s

def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(_sanitize_problematic_smiles(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _remove_atom_map(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol.GetMol()


@lru_cache(maxsize=None)
def canonicalize_smiles(smiles: str) -> str:
    """Canonical SMILES with atom-map numbers stripped."""
    mol = _mol_from_smiles(smiles)
    mol = _remove_atom_map(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=None)
def _charge_parent_smiles(smiles: str) -> str:
    """ChargeParent only (no fragment removal). Cached per unique SMILES."""
    mol = _mol_from_smiles(smiles)
    mol = rdMolStandardize.ChargeParent(mol)
    return Chem.MolToSmiles(mol)


def _parse_formula(formula: str):
    """Return (heavy_atom_str, h_count, charge_sign) from a molecular formula string.

    Charge sign is +1 for cations (formula ends with '+'), -1 for anions ('-'), 0 otherwise.
    """
    charge_sign = 0
    if formula.endswith("+"):
        charge_sign = +1
    elif formula.endswith("-"):
        charge_sign = -1
    formula_no_charge = re.sub(r"[+-]+$", "", formula)
    # Extract H count explicitly (e.g. "H4B" -> 4, "CH4O" -> 4, "C2H3O2" -> 3)
    m = re.search(r"H(\d*)", formula_no_charge)
    if m:
        h_count = int(m.group(1)) if m.group(1) else 1
        heavy = formula_no_charge[:m.start()] + formula_no_charge[m.end():]
    else:
        h_count = 0
        heavy = formula_no_charge
    return heavy, h_count, charge_sign


@lru_cache(maxsize=None)
def _charge_parent_if_proton_transfer(smiles: str) -> str:
    """Apply ChargeParent only when the transform corresponds to a simple H+ transfer.

    Allowed: Δcharge and ΔH have the *same* sign (gain/lose a proton together),
             i.e. protonation (charge+1, H+1) or deprotonation (charge−1, H−1),
             and heavy-atom skeleton is unchanged.
    Blocked: Δcharge and ΔH have *opposite* signs — this indicates hydride (H−)
             transfer or other non-proton-transfer chemistry (e.g. [BH4−] → B).
    """
    parent = _charge_parent_smiles(smiles)
    if parent == smiles:
        return smiles
    try:
        src_formula = rdMolDescriptors.CalcMolFormula(_mol_from_smiles(smiles))
        dst_formula = rdMolDescriptors.CalcMolFormula(_mol_from_smiles(parent))
    except Exception:
        return smiles
    src_heavy, src_h, src_q = _parse_formula(src_formula)
    dst_heavy, dst_h, dst_q = _parse_formula(dst_formula)
    # Heavy-atom skeleton must be identical.
    if src_heavy != dst_heavy:
        return smiles
    delta_q = dst_q - src_q
    delta_h = dst_h - src_h
    # Allow only when ΔH and Δcharge agree in sign (pure proton transfer).
    if delta_q == 0 and delta_h == 0:
        return parent  # no change — always fine
    if delta_q * delta_h > 0 or (delta_h == 0 and delta_q != 0):
        # same sign or pure charge redistribution without H change — allow
        return parent
    # Opposite signs → hydride or other non-proton chemistry — block.
    return smiles


# ---------------------------------------------------------------------------
# Reaction-side parsing
# ---------------------------------------------------------------------------

def _parse_sides(rxn_smiles: str):
    """Return (reactant_smiles_list, product_smiles_list)."""
    parts = rxn_smiles.strip().split(">")
    if len(parts) == 3:
        r_str, _, p_str = parts
    elif len(parts) == 2:
        r_str, p_str = parts
    else:
        raise ValueError(f"Cannot parse reaction SMILES: {rxn_smiles!r}")
    reactants = [s for s in r_str.split(".") if s.strip()]
    products  = [s for s in p_str.split(".") if s.strip()]
    return reactants, products


def _proton_shuffle_counter(canon_smiles_list: List[str]) -> Counter:
    """Build proton-shuffle signature from canonicalized species list.

    Protons can move between molecules on the same side; halides are preserved
    as spectators. ChargeParent is applied only when classified as proton
    transfer by _charge_parent_if_proton_transfer.
    """
    out: Counter = Counter()
    for smi in canon_smiles_list:
        if smi == _HPLUS_CANON:
            continue
        if smi in _HALIDE_ANIONS:
            out[smi] += 1
            continue
        try:
            out[canonicalize_smiles(_charge_parent_if_proton_transfer(smi))] += 1
        except Exception:
            out[smi] += 1
    return out


@lru_cache(maxsize=None)
def _normalize_side_cached(smiles_tuple: tuple, mode: str) -> Counter:
    """
    Return a Counter of normalized SMILES for one reaction side.

    mode='canonical'
        Pure canonical SMILES, no EQUIV_MAP expansion.
    mode='strict' / 'equiv'
        Canonical SMILES after EQUIV_MAP expansion.  Both modes are equivalent
        and kept for backward compatibility.
    mode='ionic_equiv'
        EQUIV_MAP expansion, then configurable salt-like dissociation to
        explicit ions using the cation/anion lists in this module.
    mode='proton_shuffle'
        Alias for post-EQUIV proton shuffle (kept for backwards compatibility
        in debugging scripts).
    mode='proton_shuffle_pre_equiv'
        Proton-shuffle signature before EQUIV_MAP expansion.
    mode='proton_shuffle_post_equiv'
        Proton-shuffle signature after EQUIV_MAP expansion.
    """
    if mode == "canonical":
        return Counter(canonicalize_smiles(smi) for smi in smiles_tuple)

    canonicalized = [canonicalize_smiles(smi) for smi in smiles_tuple]

    if mode == "proton_shuffle_pre_equiv":
        return _proton_shuffle_counter(canonicalized)

    # Step 0 (equiv/proton_shuffle_post_equiv): expand EQUIV_MAP entries.
    expanded: List[str] = []
    for canon in canonicalized:
        expansions = _EQUIV_MAP.get(canon)
        if expansions is not None:
            expanded.extend(expansions)
        else:
            expanded.append(canon)

    if mode in ("strict", "equiv"):
        return Counter(expanded)

    if mode == "ionic_equiv":
        ionic: Counter = Counter()
        for smi in expanded:
            ionic += _dissociate_ionic_species(smi)
        return ionic

    if mode in ("proton_shuffle", "proton_shuffle_post_equiv"):
        return _proton_shuffle_counter(expanded)

    raise ValueError(f"Unknown mode: {mode!r}")


def _normalize_side(smiles_list: List[str], mode: str) -> Counter:
    """Thin wrapper — converts list to tuple so the cached inner function can key on it."""
    return _normalize_side_cached(tuple(smiles_list), mode)


def _counter_signature(counter: Counter) -> tuple:
    """Hashable stoichiometry-aware signature for normalized side counters."""
    return tuple(sorted(counter.items()))


def _set_signature(counter: Counter) -> tuple:
    """Hashable stoichiometry-agnostic signature derived from counter keys."""
    return tuple(sorted(counter.keys()))


def _all_side_signatures(smiles_list: List[str]) -> set:
    """
    Return signatures for all currently supported normalization pathways.

    Includes stoichiometry-aware and set-based proton-shuffle views.
    """
    sigs: set = set()
    canonical = _normalize_side(smiles_list, "canonical")
    equiv = _normalize_side(smiles_list, "equiv")
    ionic = _normalize_side(smiles_list, "ionic_equiv")
    ps_pre = _normalize_side(smiles_list, "proton_shuffle_pre_equiv")
    ps_post = _normalize_side(smiles_list, "proton_shuffle_post_equiv")

    for c in (canonical, equiv, ionic, ps_pre, ps_post):
        sigs.add(("counter", _counter_signature(c)))
    for c in (ps_pre, ps_post):
        sigs.add(("set", _set_signature(c)))
    return sigs


def _element_counter(mol: Chem.Mol) -> Counter:
    """Count elemental composition of a molecule."""
    out: Counter = Counter()
    for atom in mol.GetAtoms():
        out[atom.GetSymbol()] += 1
    return out


def _molecule_formal_charge(mol: Chem.Mol) -> int:
    """Total formal charge for a molecule."""
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


# ---------------------------------------------------------------------------
# Small-molecule equivalence table
# Populated after helper functions are defined (see bottom of section).
# ---------------------------------------------------------------------------

def _c(s: str) -> str:
    """Canonicalize at import time for the equivalence table."""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError(f"Bad SMILES in equivalence table: {s!r}")
    return Chem.MolToSmiles(mol)


# Maps one canonical SMILES → list of canonical SMILES it expands to.
# Only small molecules with well-known dual representations are listed.
_EQUIV_MAP: dict = {}
_HALIDE_ANIONS = {
    canonicalize_smiles("[Cl-]"),
    canonicalize_smiles("[Br-]"),
    canonicalize_smiles("[I-]"),
    canonicalize_smiles("[F-]"),
}
_HPLUS_CANON = canonicalize_smiles("[H+]")
_H2O_CANON = canonicalize_smiles("O")
_HMINUS_CANON = canonicalize_smiles("[H-]")
_HATOM_CANON = canonicalize_smiles("[H]")
_OATOM_CANON = canonicalize_smiles("[O]")
_H2_CANON = canonicalize_smiles("[H][H]")
_OHM_CANON = canonicalize_smiles("[OH-]")
_SO2_CANON = canonicalize_smiles("O=S=O")
_HSOCL_CANON = canonicalize_smiles("O=S(O)Cl")
_BH2NA_CANON = canonicalize_smiles("[BH2][Na]")
_BH4_CANON = canonicalize_smiles("[BH4-]")
_NAPLUS_CANON = canonicalize_smiles("[Na+]")


def _deep_pragmatic_side_counter(smiles_list: List[str]) -> Counter:
    """Normalize one side with extra rewrite rules for common representation artifacts."""
    ctr = Counter(_normalize_side(smiles_list, "ionic_equiv"))

    # Hydrolysis shorthand: chlorosulfurous acid <-> sulfur dioxide + H+ + Cl-
    n = int(ctr.get(_HSOCL_CANON, 0))
    if n > 0:
        ctr[_HSOCL_CANON] -= n
        if ctr[_HSOCL_CANON] <= 0:
            del ctr[_HSOCL_CANON]
        ctr[_SO2_CANON] += n
        ctr[_HPLUS_CANON] += n
        ctr[canonicalize_smiles("[Cl-]")] += n

    # Borohydride shorthand often appears as [BH2][Na] in predictions.
    n = int(ctr.get(_BH2NA_CANON, 0))
    if n > 0:
        ctr[_BH2NA_CANON] -= n
        if ctr[_BH2NA_CANON] <= 0:
            del ctr[_BH2NA_CANON]
        ctr[_BH4_CANON] += n
        ctr[_NAPLUS_CANON] += n

    # Normalize hydrogen bookkeeping to H2 where possible.
    n = min(int(ctr.get(_HPLUS_CANON, 0)), int(ctr.get(_HMINUS_CANON, 0)))
    if n > 0:
        ctr[_HPLUS_CANON] -= n
        ctr[_HMINUS_CANON] -= n
        if ctr[_HPLUS_CANON] <= 0:
            del ctr[_HPLUS_CANON]
        if ctr[_HMINUS_CANON] <= 0:
            del ctr[_HMINUS_CANON]
        ctr[_H2_CANON] += n

    n = int(ctr.get(_HATOM_CANON, 0)) // 2
    if n > 0:
        ctr[_HATOM_CANON] -= 2 * n
        if ctr[_HATOM_CANON] <= 0:
            del ctr[_HATOM_CANON]
        ctr[_H2_CANON] += n

    # Canonical acid/base neutralizations used in datasets.
    n = min(int(ctr.get(_HPLUS_CANON, 0)), int(ctr.get(_OHM_CANON, 0)))
    if n > 0:
        ctr[_HPLUS_CANON] -= n
        ctr[_OHM_CANON] -= n
        if ctr[_HPLUS_CANON] <= 0:
            del ctr[_HPLUS_CANON]
        if ctr[_OHM_CANON] <= 0:
            del ctr[_OHM_CANON]
        ctr[_H2O_CANON] += n

    for halide in _HALIDE_ANIONS:
        n = min(int(ctr.get(_HPLUS_CANON, 0)), int(ctr.get(halide, 0)))
        if n > 0:
            ctr[_HPLUS_CANON] -= n
            ctr[halide] -= n
            if ctr[_HPLUS_CANON] <= 0:
                del ctr[_HPLUS_CANON]
            if ctr[halide] <= 0:
                del ctr[halide]
            mol = {canonicalize_smiles("[Cl-]"): "Cl", canonicalize_smiles("[Br-]"): "Br", canonicalize_smiles("[I-]"): "I", canonicalize_smiles("[F-]"): "F"}[halide]
            ctr[canonicalize_smiles(mol)] += n

    return ctr

# ---------------------------------------------------------------------------
# Ionic dissociation configuration (chemist-editable)
# ---------------------------------------------------------------------------
# Cation species that may represent dissolved ions in ionic notation.
_IONIC_CATION_SPECIES = [
    "[Li+]", "[Na+]", "[K+]", "[Rb+]", "[Cs+]", "[NH4+]",
    "[Mg+2]", "[Ca+2]", "[Sr+2]", "[Ba+2]", "[Zn+2]", "[Cu+2]", "[Ni+2]", "[Co+2]",
    "[Fe+2]", "[Fe+3]", "[Al+3]", "[Cr+3]",
]

# Anion species that may represent dissolved ions in ionic notation.
_IONIC_ANION_SPECIES = [
    "[F-]", "[Cl-]", "[Br-]", "[I-]", "[OH-]",
    "[O-]S(=O)(=O)[O-]",  # sulfate
    "O=S(=O)([O-])O",      # bisulfate
    "[O-]N(=O)=O",         # nitrate
    "[O-]C(=O)O",          # bicarbonate
    "[O-]C([O-])=O",       # carbonate
    "CC(=O)[O-]",          # acetate
    "[O-]P(=O)([O-])[O-]", # phosphate
]


def _compile_ion_specs(smiles_list: List[str]) -> List[Tuple[str, int, Counter]]:
    """Compile ion templates to (canonical_smiles, formal_charge, element_counter)."""
    out: List[Tuple[str, int, Counter]] = []
    for smi in smiles_list:
        canon = canonicalize_smiles(smi)
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            continue
        charge = _molecule_formal_charge(mol)
        out.append((canon, charge, _element_counter(mol)))
    return out


_CATION_ION_SPECS = [x for x in _compile_ion_specs(_IONIC_CATION_SPECIES) if x[1] > 0]
_ANION_ION_SPECS = [x for x in _compile_ion_specs(_IONIC_ANION_SPECIES) if x[1] < 0]


def _dissociate_ionic_species(canon_smiles: str) -> Counter:
    """
    Convert a neutral salt-like species into configured ions when plausible.

    If no plausible dissociation is found, returns Counter({canon_smiles: 1}).
    """
    if canon_smiles in _HALIDE_ANIONS or canon_smiles == _HPLUS_CANON:
        return Counter({canon_smiles: 1})

    mol = Chem.MolFromSmiles(canon_smiles)
    if mol is None:
        return Counter({canon_smiles: 1})

    if _molecule_formal_charge(mol) != 0:
        # Already explicitly ionic.
        return Counter({canon_smiles: 1})

    elem = _element_counter(mol)

    for c_smi, q_c, c_elem in _CATION_ION_SPECS:
        for a_smi, q_a, a_elem in _ANION_ION_SPECS:
            # Integer stoichiometry that neutralizes charge.
            gcd_qa_qc = int(np.gcd(abs(q_a), q_c))
            n_c = abs(q_a) // gcd_qa_qc
            n_a = q_c // gcd_qa_qc

            reconstructed: Counter = Counter()
            for k, v in c_elem.items():
                reconstructed[k] += n_c * v
            for k, v in a_elem.items():
                reconstructed[k] += n_a * v

            if reconstructed == elem:
                return Counter({c_smi: n_c, a_smi: n_a})

    return Counter({canon_smiles: 1})

def _build_equiv_map():
    CO2    = _c("O=C=O")
    H2O    = _c("O")
    H2CO3  = _c("OC(=O)O")
    H3Op   = _c("[OH3+]")
    Hplus  = _c("[H+]")
    OHm    = _c("[OH-]")
    Cl_m   = _c("[Cl-]")
    Br_m   = _c("[Br-]")
    I_m    = _c("[I-]")
    HCl    = _c("Cl")
    HBr    = _c("Br")
    HI     = _c("I")
    Naplus = _c("[Na+]")
    Kplus  = _c("[K+]")
    Liplus = _c("[Li+]")
    NaOH   = _c("[OH][Na]")   # canonical: [OH][Na]
    KOH    = _c("[OH][K]")    # canonical: [OH][K]
    LiOH   = _c("[OH][Li]")   # canonical: [Li][OH]
    NaH    = _c("[NaH]")
    KH     = _c("[KH]")
    LiH    = _c("[LiH]")
    NaI    = _c("[Na]I")
    KI     = _c("[K]I")
    LiI    = _c("[Li]I")
    NaOMe  = _c("C[O][Na]")
    NaOEt  = _c("CC[O][Na]")
    KOMe   = _c("C[O][K]")
    KOEt   = _c("CC[O][K]")
    LiOMe  = _c("C[O][Li]")
    LiOEt  = _c("CC[O][Li]")
    BH4Na1 = _c("[BH4-].[Na+]")
    BH4Na2 = _c("[BH2][Na].[H].[H]")
    SOCl2  = _c("O=S(Cl)Cl")
    SO2    = _c("O=S=O")
    _EQUIV_MAP[H2CO3] = [CO2, H2O]
    _EQUIV_MAP[H3Op]  = [Hplus, H2O]
    _EQUIV_MAP[HCl]   = [Hplus, Cl_m]
    _EQUIV_MAP[HBr]   = [Hplus, Br_m]
    _EQUIV_MAP[HI]    = [Hplus, I_m]
    _EQUIV_MAP[NaOH]  = [Naplus, OHm]
    _EQUIV_MAP[KOH]   = [Kplus,  OHm]
    _EQUIV_MAP[LiOH]  = [Liplus, OHm]
    _EQUIV_MAP[NaH]   = [Naplus, _c("[H-]")]
    _EQUIV_MAP[KH]    = [Kplus,  _c("[H-]")]
    _EQUIV_MAP[LiH]   = [Liplus, _c("[H-]")]
    _EQUIV_MAP[NaI]   = [Naplus, I_m]
    _EQUIV_MAP[KI]    = [Kplus,  I_m]
    _EQUIV_MAP[LiI]   = [Liplus, I_m]
    _EQUIV_MAP[NaOMe] = [Naplus, _c("C[O-]")]
    _EQUIV_MAP[NaOEt] = [Naplus, _c("CC[O-]")]
    _EQUIV_MAP[KOMe]  = [Kplus,  _c("C[O-]")]
    _EQUIV_MAP[KOEt]  = [Kplus,  _c("CC[O-]")]
    _EQUIV_MAP[LiOMe] = [Liplus, _c("C[O-]")]
    _EQUIV_MAP[LiOEt] = [Liplus, _c("CC[O-]")]
    _EQUIV_MAP[BH4Na2] = [BH4Na1]
    # Hydrolysis shorthand often appears as SOCl2 + H2O vs SO2 + 2 HCl.
    _EQUIV_MAP[SOCl2] = [SO2, HCl, HCl]
    # symmetric: ions → molecular form
    # (not added by default; reactions rarely write H⁺ + Cl⁻ in GT)

_build_equiv_map()


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """
    Per-pair comparison result at five levels of strictness.

    Attributes
    ----------
    exact_match:
        Pure canonical SMILES sets (with stoichiometry).  No EQUIV_MAP
        expansion, no standardization.  Strictest level.
    proton_shuffle_match:
        Proton-transfer-aware normalization is applied both before and after
        EQUIV_MAP expansion. Match passes if either pathway matches (free H+
        is mobile; halides preserved; other species reduced with ChargeParent).
    all_normalizations_match:
        Aggregate mode: compute all normalization signatures on both predicted
        and ground-truth sides and compare all-against-all. Match passes when
        reactant signatures overlap and product signatures overlap.
    pragmatic_equivalence_match:
        Catch-all metric that combines all-normalizations overlap,
        difference-normalized matching, and a same-path comparison after
        cancelling spectators within each reaction side pair.
    deep_pragmatic_equivalence_match:
        Pragmatic metric plus deeper, targeted rewrite normalization before
        side comparison, intended for robust benchmarking with heterogeneous
        reaction/byproduct notation.
    """
    exact_match: bool
    proton_shuffle_match: bool
    all_normalizations_match: bool
    pragmatic_equivalence_match: bool
    deep_pragmatic_equivalence_match: bool

    def any_match(self) -> bool:
        """True if at least proton_shuffle_match holds."""
        return self.proton_shuffle_match


def compare_reactions(pred_rxn: str, gt_rxn: str) -> ComparisonResult:
    """
    Compare a predicted reaction SMILES against a ground-truth reaction SMILES.

    Returns a :class:`ComparisonResult` with four boolean fields.
    Reagents (middle section of ``A>R>B`` format) are ignored.
    """
    pred_r, pred_p = _parse_sides(pred_rxn)
    gt_r,   gt_p   = _parse_sides(gt_rxn)

    exact_match = (
        _normalize_side(pred_r, "canonical") == _normalize_side(gt_r, "canonical")
        and _normalize_side(pred_p, "canonical") == _normalize_side(gt_p, "canonical")
    )
    proton_shuffle_pre_match = (
        _normalize_side(pred_r, "proton_shuffle_pre_equiv") == _normalize_side(gt_r, "proton_shuffle_pre_equiv")
        and _normalize_side(pred_p, "proton_shuffle_pre_equiv") == _normalize_side(gt_p, "proton_shuffle_pre_equiv")
    )
    proton_shuffle_post_match = (
        _normalize_side(pred_r, "proton_shuffle_post_equiv") == _normalize_side(gt_r, "proton_shuffle_post_equiv")
        and _normalize_side(pred_p, "proton_shuffle_post_equiv") == _normalize_side(gt_p, "proton_shuffle_post_equiv")
    )
    proton_shuffle_match = proton_shuffle_pre_match or proton_shuffle_post_match

    pred_r_pre_set = set(_normalize_side(pred_r, "proton_shuffle_pre_equiv").keys())
    gt_r_pre_set = set(_normalize_side(gt_r, "proton_shuffle_pre_equiv").keys())
    pred_p_pre_set = set(_normalize_side(pred_p, "proton_shuffle_pre_equiv").keys())
    gt_p_pre_set = set(_normalize_side(gt_p, "proton_shuffle_pre_equiv").keys())
    proton_shuffle_pre_set_match = (pred_r_pre_set == gt_r_pre_set) and (pred_p_pre_set == gt_p_pre_set)

    pred_r_post_set = set(_normalize_side(pred_r, "proton_shuffle_post_equiv").keys())
    gt_r_post_set = set(_normalize_side(gt_r, "proton_shuffle_post_equiv").keys())
    pred_p_post_set = set(_normalize_side(pred_p, "proton_shuffle_post_equiv").keys())
    gt_p_post_set = set(_normalize_side(gt_p, "proton_shuffle_post_equiv").keys())
    proton_shuffle_post_set_match = (pred_r_post_set == gt_r_post_set) and (pred_p_post_set == gt_p_post_set)

    pred_r_sigs = _all_side_signatures(pred_r)
    gt_r_sigs = _all_side_signatures(gt_r)
    pred_p_sigs = _all_side_signatures(pred_p)
    gt_p_sigs = _all_side_signatures(gt_p)
    all_normalizations_match = bool(pred_r_sigs & gt_r_sigs) and bool(pred_p_sigs & gt_p_sigs)

    def _counter_to_list(counter: Counter) -> List[str]:
        out: List[str] = []
        for smi, n in counter.items():
            out.extend([smi] * int(n))
        return out

    def _counter_minus(a: Counter, b: Counter) -> Counter:
        out: Counter = Counter()
        for k, v in a.items():
            rem = int(v) - int(b.get(k, 0))
            if rem > 0:
                out[k] = rem
        return out

    def _cancel_between_sides(r_ctr: Counter, p_ctr: Counter) -> Tuple[Counter, Counter]:
        r_out = Counter(r_ctr)
        p_out = Counter(p_ctr)
        for smi in list(r_out.keys()):
            c = min(int(r_out.get(smi, 0)), int(p_out.get(smi, 0)))
            if c > 0:
                r_out[smi] -= c
                p_out[smi] -= c
                if r_out[smi] <= 0:
                    del r_out[smi]
                if p_out[smi] <= 0:
                    del p_out[smi]
        return r_out, p_out

    def _cancel_species_between_counters(a: Counter, b: Counter, smi: str) -> Tuple[Counter, Counter]:
        a_out = Counter(a)
        b_out = Counter(b)
        c = min(int(a_out.get(smi, 0)), int(b_out.get(smi, 0)))
        if c > 0:
            a_out[smi] -= c
            b_out[smi] -= c
            if a_out[smi] <= 0:
                del a_out[smi]
            if b_out[smi] <= 0:
                del b_out[smi]
        return a_out, b_out

    # 1) Canonical side counters
    pred_r_can = _normalize_side(pred_r, "canonical")
    gt_r_can = _normalize_side(gt_r, "canonical")
    pred_p_can = _normalize_side(pred_p, "canonical")
    gt_p_can = _normalize_side(gt_p, "canonical")

    # 2) Keep only unmatched molecules per side between prediction and target
    pred_r_diff = _counter_minus(pred_r_can, gt_r_can)
    gt_r_diff = _counter_minus(gt_r_can, pred_r_can)
    pred_p_diff = _counter_minus(pred_p_can, gt_p_can)
    gt_p_diff = _counter_minus(gt_p_can, pred_p_can)

    # 3) Cancel spectators that appear on both sides of each difference reaction.
    #    This makes extra H2O on both left/right a valid no-op transformation.
    pred_r_diff, pred_p_diff = _cancel_between_sides(pred_r_diff, pred_p_diff)
    gt_r_diff, gt_p_diff = _cancel_between_sides(gt_r_diff, gt_p_diff)

    # 3b) Also allow side-flipped spectator water between prediction and target
    #     (reactant <-> product), e.g., water written on opposite sides.
    pred_r_diff, gt_p_diff = _cancel_species_between_counters(pred_r_diff, gt_p_diff, _H2O_CANON)
    pred_p_diff, gt_r_diff = _cancel_species_between_counters(pred_p_diff, gt_r_diff, _H2O_CANON)

    # 4) Compare only normalized difference signatures (all pathways, combinable).
    pred_r_diff_sigs = _all_side_signatures(_counter_to_list(pred_r_diff))
    gt_r_diff_sigs = _all_side_signatures(_counter_to_list(gt_r_diff))
    pred_p_diff_sigs = _all_side_signatures(_counter_to_list(pred_p_diff))
    gt_p_diff_sigs = _all_side_signatures(_counter_to_list(gt_p_diff))
    difference_normalized_match = bool(pred_r_diff_sigs & gt_r_diff_sigs) and bool(pred_p_diff_sigs & gt_p_diff_sigs)

    # Compare reaction cores after cancelling shared spectators on each side.
    def _core_match_same_path(mode: str, set_view: bool = False) -> bool:
        pr = _normalize_side(pred_r, mode)
        pp = _normalize_side(pred_p, mode)
        gr = _normalize_side(gt_r, mode)
        gp = _normalize_side(gt_p, mode)
        pr, pp = _cancel_between_sides(pr, pp)
        gr, gp = _cancel_between_sides(gr, gp)
        if set_view:
            return (set(pr.keys()) == set(gr.keys())) and (set(pp.keys()) == set(gp.keys()))
        return (pr == gr) and (pp == gp)

    pragmatic_equivalence_match = (
        all_normalizations_match
        or difference_normalized_match
        or _core_match_same_path("canonical")
        or _core_match_same_path("equiv")
        or _core_match_same_path("ionic_equiv")
        or _core_match_same_path("proton_shuffle_pre_equiv")
        or _core_match_same_path("proton_shuffle_post_equiv")
        or _core_match_same_path("proton_shuffle_pre_equiv", set_view=True)
        or _core_match_same_path("proton_shuffle_post_equiv", set_view=True)
    )

    # Deep pragmatic metric with additional targeted rewrite layer.
    pred_r_deep = _deep_pragmatic_side_counter(pred_r)
    pred_p_deep = _deep_pragmatic_side_counter(pred_p)
    gt_r_deep = _deep_pragmatic_side_counter(gt_r)
    gt_p_deep = _deep_pragmatic_side_counter(gt_p)

    deep_direct = (pred_r_deep == gt_r_deep) and (pred_p_deep == gt_p_deep)
    deep_set = (set(pred_r_deep.keys()) == set(gt_r_deep.keys())) and (set(pred_p_deep.keys()) == set(gt_p_deep.keys()))

    pred_r_deep_core, pred_p_deep_core = _cancel_between_sides(pred_r_deep, pred_p_deep)
    gt_r_deep_core, gt_p_deep_core = _cancel_between_sides(gt_r_deep, gt_p_deep)
    deep_core = (pred_r_deep_core == gt_r_deep_core) and (pred_p_deep_core == gt_p_deep_core)
    deep_core_set = (set(pred_r_deep_core.keys()) == set(gt_r_deep_core.keys())) and (set(pred_p_deep_core.keys()) == set(gt_p_deep_core.keys()))

    deep_pragmatic_equivalence_match = pragmatic_equivalence_match or deep_direct or deep_set or deep_core or deep_core_set

    # Guardrail: do not score reactions as accurate when prediction contains
    # extra standalone atomic oxygen ([O]) not present in ground truth.
    pred_oatoms = int(pred_r_can.get(_OATOM_CANON, 0)) + int(pred_p_can.get(_OATOM_CANON, 0))
    gt_oatoms = int(gt_r_can.get(_OATOM_CANON, 0)) + int(gt_p_can.get(_OATOM_CANON, 0))
    if pred_oatoms > gt_oatoms:
        proton_shuffle_match = False
        all_normalizations_match = False
        pragmatic_equivalence_match = False
        deep_pragmatic_equivalence_match = False

    return ComparisonResult(
        exact_match=exact_match,
        proton_shuffle_match=proton_shuffle_match,
        all_normalizations_match=all_normalizations_match,
        pragmatic_equivalence_match=pragmatic_equivalence_match,
        deep_pragmatic_equivalence_match=deep_pragmatic_equivalence_match,
    )


def _proton_roles_for_smiles(smiles: str) -> Dict[str, int]:
    """Return simple donor/acceptor counts used for proton-mobility introspection."""
    if smiles == _HPLUS_CANON:
        return {"donor_sites": 1, "acceptor_sites": 0}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"donor_sites": 0, "acceptor_sites": 0}
    return {
        "donor_sites": int(Lipinski.NumHDonors(mol)),
        "acceptor_sites": int(Lipinski.NumHAcceptors(mol)),
    }


def proton_transfer_dictionary(rxn_smiles: str) -> Dict[str, Dict[str, List[dict]]]:
    """
    Inspect one reaction and report where protons can plausibly move.

    Returns a dictionary for reactants/products with species-level proton
    donor/acceptor roles and pairwise donor→acceptor candidates.
    """
    reactants, products = _parse_sides(rxn_smiles)

    def _expand(smiles_list: List[str]) -> Counter:
        expanded: List[str] = []
        for smi in smiles_list:
            canon = canonicalize_smiles(smi)
            expansions = _EQUIV_MAP.get(canon)
            if expansions is not None:
                expanded.extend(expansions)
            else:
                expanded.append(canon)
        return Counter(expanded)

    def _side_dict(smiles_list: List[str]) -> Dict[str, List[dict]]:
        expanded = _expand(smiles_list)
        species: List[dict] = []
        donors: List[str] = []
        acceptors: List[str] = []
        for smi, count in expanded.items():
            roles = _proton_roles_for_smiles(smi)
            species.append(
                {
                    "smiles": smi,
                    "count": int(count),
                    "donor_sites": roles["donor_sites"],
                    "acceptor_sites": roles["acceptor_sites"],
                }
            )
            if roles["donor_sites"] > 0:
                donors.append(smi)
            if roles["acceptor_sites"] > 0:
                acceptors.append(smi)

        candidates: List[dict] = []
        for donor in sorted(set(donors)):
            for acceptor in sorted(set(acceptors)):
                if donor == acceptor:
                    continue
                candidates.append({"from": donor, "to": acceptor})

        return {
            "species": species,
            "proton_transfer_candidates": candidates,
        }

    return {
        "reactants": _side_dict(reactants),
        "products": _side_dict(products),
    }


# Convenience wrapper kept for backwards compatibility with score_predictions.py
# and evaluate.py (which inline their own copy for container environments).
def reaction_set_match(pred_rxn: str, target_rxn: str) -> bool:
    """Exact canonical match (stoichiometry-aware). Alias for compare_reactions(...).exact_match."""
    try:
        return compare_reactions(pred_rxn, target_rxn).exact_match
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Aggregate accuracy helpers
# ---------------------------------------------------------------------------

def flexible_accuracy(
    predictions: List[str],
    targets: List[str],
    match_fn: Optional[Callable[[str, str], bool]] = None,
    **match_kwargs,
) -> float:
    """
    Fraction of (prediction, target) pairs that match.

    Parameters
    ----------
    predictions, targets:
        Parallel lists of reaction SMILES.
    match_fn:
        ``(pred, target) -> bool``.  Defaults to
        ``lambda p, t: compare_reactions(p, t).proton_shuffle_match``.
    """
    if not predictions and not targets:
        return 0.0
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have the same length "
            f"({len(predictions)} vs {len(targets)})"
        )
    if match_fn is None:
        match_fn = lambda p, t: compare_reactions(p, t).proton_shuffle_match
    correct = sum(match_fn(p, t, **match_kwargs) for p, t in zip(predictions, targets))
    return correct / len(predictions)


def top_k_accuracy(
    beam_predictions: List[List[str]],
    targets: List[str],
    k: int = 1,
    match_fn: Optional[Callable[[str, str], bool]] = None,
    **match_kwargs,
) -> float:
    """
    Top-*k* accuracy over beam predictions.

    A sample is correct if any of its first *k* candidates matches the target.

    Parameters
    ----------
    beam_predictions:
        ``beam_predictions[i]`` is the ranked candidate list for sample *i*.
    targets:
        Ground-truth reaction SMILES.
    k:
        Beam depth to consider.
    match_fn:
        ``(pred, target) -> bool``.  Defaults to proton_shuffle_match.
    """
    if not beam_predictions and not targets:
        return 0.0
    if len(beam_predictions) != len(targets):
        raise ValueError(
            f"beam_predictions and targets must have the same length "
            f"({len(beam_predictions)} vs {len(targets)})"
        )
    if match_fn is None:
        match_fn = lambda p, t: compare_reactions(p, t).proton_shuffle_match
    correct = sum(
        any(match_fn(p, tgt, **match_kwargs) for p in preds[:k])
        for preds, tgt in zip(beam_predictions, targets)
    )
    return correct / len(beam_predictions)


# ---------------------------------------------------------------------------
# Reaction similarity (DrFP-style differential ECFP Tanimoto)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _smi_ecfp(smi: str, radius: int, n_bits: int) -> np.ndarray:
    """ECFP bit-vector for one SMILES. Cached per (smi, radius, n_bits)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _side_fp(smiles_list: List[str], radius: int, n_bits: int) -> np.ndarray:
    """OR-combine per-molecule ECFP bit vectors for a reaction side."""
    combined = np.zeros(n_bits, dtype=np.uint8)
    for smi in smiles_list:
        combined |= _smi_ecfp(smi, radius, n_bits)
    return combined


def reaction_similarity(
    rxn1: str,
    rxn2: str,
    method: str = "ecfp",
    radius: int = 3,
    n_bits: int = 2048,
    ignore_reagents: bool = True,
) -> float:
    """
    Continuous similarity in [0, 1] between two reaction SMILES.

    method='ecfp'
        Differential Morgan fingerprint Tanimoto (DrFP-style): per-molecule
        ECFP bit vectors are OR-combined per side, XOR-ed to get the
        differential fingerprint, and Tanimoto is computed between the two.
    method='set_match'
        Binary exact match via :func:`reaction_set_match` (returns 0.0 or 1.0).
    """
    if method == "set_match":
        return 1.0 if reaction_set_match(rxn1, rxn2) else 0.0

    if method == "ecfp":
        try:
            r1, p1 = _parse_sides(rxn1)
            r2, p2 = _parse_sides(rxn2)
        except ValueError:
            return 0.0
        diff1 = _side_fp(r1, radius, n_bits) ^ _side_fp(p1, radius, n_bits)
        diff2 = _side_fp(r2, radius, n_bits) ^ _side_fp(p2, radius, n_bits)
        inter = int(np.sum(diff1 & diff2))
        union = int(np.sum(diff1 | diff2))
        return inter / union if union else 0.0

    raise ValueError(f"Unknown method: {method!r}. Choose 'ecfp' or 'set_match'.")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _check(label, result, expect_exact, expect_equiv, expect_ps=None):
        ok_e = result.exact_match == expect_exact
        ok_q = result.equivalence_match == expect_equiv
        ok_p = (result.proton_shuffle_match == expect_ps) if expect_ps is not None else True
        status = "PASS" if (ok_e and ok_q and ok_p) else "FAIL"
        print(f"[{status}] {label}")
        if status == "FAIL":
            print(f"       exact:          got {result.exact_match}, expected {expect_exact}")
            print(f"       equivalence:    got {result.equivalence_match}, expected {expect_equiv}")
            if expect_ps is not None:
                print(f"       proton_shuffle: got {result.proton_shuffle_match}, expected {expect_ps}")

    # ------------------------------------------------------------------
    # 1. Identical reactions — all levels match
    # ------------------------------------------------------------------
    _check(
        "Identical (ethanol oxidation)",
        compare_reactions("CCO>>CC=O", "CCO>>CC=O"),
        expect_exact=True, expect_equiv=True, expect_ps=True,
    )

    # ------------------------------------------------------------------
    # 2. Molecule order on reactant side — Counter is order-invariant
    # ------------------------------------------------------------------
    _check(
        "Molecule order invariance (A.B>>C vs B.A>>C)",
        compare_reactions("CCO.O>>CC=O.O", "O.CCO>>CC=O.O"),
        expect_exact=True, expect_equiv=True, expect_ps=True,
    )

    # ------------------------------------------------------------------
    # 3. H₂CO₃ ↔ CO₂ + H₂O  (carbonate ester ring-opening)
    #    A model predicts CO2 + H2O; ground truth writes H2CO3.
    #    exact fails (no EQUIV_MAP); equivalence_match and proton_shuffle must be True.
    # ------------------------------------------------------------------
    rxn_co2 = (
        "C=CC1(c2ccc(C)cc2)COC(=O)O1."
        "Cc1ccc(S(=O)(=O)N=C2C(=Cc3ccccc3)Oc3ccccc32)cc1>>"
        "Cc1ccc(C2=CCN(S(=O)(=O)c3ccc(C)cc3)c3c(oc4ccccc34)"
        "C(c3ccccc3)OC2)cc1.O=C=O.O"
    )
    rxn_h2co3 = (
        "C=CC1(c2ccc(C)cc2)COC(=O)O1."
        "Cc1ccc(S(=O)(=O)N=C2C(=Cc3ccccc3)Oc3ccccc32)cc1>>"
        "Cc1ccc(C2=CCN(S(=O)(=O)c3ccc(C)cc3)c3c(oc4ccccc34)"
        "C(c3ccccc3)OC2)cc1.OC(=O)O"
    )
    _check(
        "H2CO3 ↔ CO2 + H2O (carbonate ring-opening)",
        compare_reactions(rxn_co2, rxn_h2co3),
        expect_exact=False, expect_equiv=True, expect_ps=True,
    )

    # ------------------------------------------------------------------
    # 4. HCl ↔ H⁺ + Cl⁻  (Fischer esterification acid byproduct)
    #    exact fails (no EQUIV_MAP); equivalence_match and proton_shuffle must be True.
    # ------------------------------------------------------------------
    _check(
        "HCl ↔ H⁺ + Cl⁻ (acid byproduct)",
        compare_reactions(
            "CCO.CC(=O)Cl>>CC(=O)OCC.[H+].[Cl-]",
            "CCO.CC(=O)Cl>>CC(=O)OCC.Cl",
        ),
        expect_exact=False, expect_equiv=True, expect_ps=True,
    )

    # ------------------------------------------------------------------
    # 5. Stoichiometry mismatch — all levels fail
    # ------------------------------------------------------------------
    _check(
        "Stoichiometry mismatch (A.A>>B vs A>>B)",
        compare_reactions("CCO.CCO>>CC=O", "CCO>>CC=O"),
        expect_exact=False, expect_equiv=False, expect_ps=False,
    )

    # ------------------------------------------------------------------
    # 6. Genuinely different reactions — all levels fail
    # ------------------------------------------------------------------
    _check(
        "Different reaction",
        compare_reactions("CCO>>CC=O", "CCO>>CCOCC"),
        expect_exact=False, expect_equiv=False, expect_ps=False,
    )

    # ------------------------------------------------------------------
    # 7. Proton shuffle: NaOH (covalent [OH][Na]) vs Na⁺+OH⁻, PLUS
    #    a proton redistributed between ethyl sulphate and bisulphate.
    #    exact/equivalence_match fail; proton_shuffle_match must be True.
    # ------------------------------------------------------------------
    completed = (
        "CCOS(=O)(=O)OCC.CSc1nc(O)cc(O)n1.[OH][Na].[Na+].O=S(=O)([O-])[O-].[Na+]>>"
        "CCOc1cc(O)nc(SC)n1.CCOS(=O)(=O)O.[OH][Na].O=S(=O)([O-])[O-].[Na+].[Na+]"
    )
    ground_truth = (
        "CCOS(=O)(=O)OCC.CSc1nc(O)cc(O)n1.O=S(=O)([O-])[O-].[Na+].[Na+].[Na+].[OH-]>>"
        "CCOS(=O)(=O)[O-].CCOc1cc(O)nc(SC)n1.O=S(=O)([O-])O.[Na+].[Na+].[Na+].[OH-]"
    )
    _check(
        "Proton shuffle: [OH][Na] vs Na⁺+OH⁻, sulphate vs bisulphate",
        compare_reactions(completed, ground_truth),
        expect_exact=False, expect_equiv=False, expect_ps=True,
    )

    # ------------------------------------------------------------------
    # 8. reaction_similarity sanity check
    # ------------------------------------------------------------------
    sim_same = reaction_similarity("CCO>>CC=O", "CCO>>CC=O")
    sim_diff = reaction_similarity("CCO>>CC=O", "c1ccccc1>>c1ccccc1O")
    assert sim_same == 1.0, f"Self-similarity should be 1.0, got {sim_same}"
    assert 0.0 <= sim_diff < sim_same, f"Unrelated reactions should score lower"
    print(f"[PASS] reaction_similarity: same={sim_same:.3f}, diff={sim_diff:.3f}")

