import argparse
import os
import subprocess
import yaml
import glob
import datetime
from pathlib import Path
from boltzgen.model.modules.material_layout import expand_material_spec, write_layout
try:
    from boltzgen.task.filter.material_presets import format_metrics_override, format_additional_filters
except ModuleNotFoundError as exc:
    # The presets module is optional in older/source-only BoltzGen installs.
    # Topologies without a custom ranking preset (such as cyclic) can still use
    # the filtering defaults without it.
    if exc.name != "boltzgen.task.filter.material_presets":
        raise

    def format_metrics_override(topology):
        # 2.0A is way too strict for multi-chain assemblies; 5.0A is more reasonable for global topology
        return "refolding_rmsd_threshold=5.0"
    def format_additional_filters(topology, *args, **kwargs):
        return None
try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx
except ImportError:
    print("Biotite not found. Please ensure it's installed (pip install biotite).")

def get_chain_id(idx):
    res = ""
    while idx >= 0:
        res = chr(65 + (idx % 26)) + res
        idx = idx // 26 - 1
    return res

def generate_yaml_from_spec(
    num_copies,
    asym_unit_def,
    output_file="material_spec.yaml",
    *,
    heteromer_counter_screen=False,
    topology=None,
    topology_params=None,
    charge_bias_strength=None,
):
    entities, layout = expand_material_spec(
        num_copies,
        asym_unit_def,
        heteromer_screening=heteromer_counter_screen,
        topology=topology,
        topology_params=topology_params,
        charge_bias_strength=charge_bias_strength,
    )
    spec = {"entities": entities}
    with open(output_file, "w") as f:
        yaml.dump(spec, f, sort_keys=False)
    output_path = Path(output_file).resolve()
    layout_path = output_path.with_name("material_layout.json")
    write_layout(layout, layout_path)
    os.environ["MAT_LAYOUT_FILE"] = str(layout_path)
    print(f"Generated design spec: {output_file}")
    return output_file

def calculate_bsa(structure_file):
    if structure_file.endswith(".cif"):
        cif_file = pdbx.CIFFile.read(structure_file)
        array = pdbx.get_structure(cif_file, model=1)
    else:
        pdb_file = pdb.PDBFile.read(structure_file)
        array = pdb.get_structure(pdb_file, model=1)
        
    # Filter for protein heavy atoms only
    protein = array[struc.filter_amino_acids(array) & (array.element != "H")]
    
    # Calculate SASA for the complex
    sasa_complex = struc.sasa(protein)
    total_sasa_complex = sasa_complex.sum()
    
    # Calculate SASA for individual chains
    total_sasa_chains = 0
    chain_ids = set(protein.chain_id)
    for c_id in chain_ids:
        chain_atoms = protein[protein.chain_id == c_id]
        sasa_chain = struc.sasa(chain_atoms)
        total_sasa_chains += sasa_chain.sum()
        
    bsa = total_sasa_chains - total_sasa_complex
    return bsa, total_sasa_complex

def main():
    parser = argparse.ArgumentParser(description="Build peptide materials using BoltzGen")
    parser.add_argument("--config", type=str, default=None, help="YAML configuration file for the material")
    parser.add_argument("--copies", type=int, default=4, help="Number of identical peptide chains")
    parser.add_argument("--length", type=int, default=15, help="Length of the peptide")
    parser.add_argument("--output_dir", type=str, default="material_out", help="Directory for BoltzGen outputs")
    parser.add_argument("--num_designs", type=int, default=1, help="Number of design candidates to generate")
    parser.add_argument("--diffusion_batch_size", type=int, default=1, help="Number of diffusion samples per trunk run (default: 1)")
    parser.add_argument("--asym_unit_size", type=int, default=1, help="Number of chains in the asymmetric unit")
    parser.add_argument("--topology", type=str, choices=["floating", "cyclic", "cylindrical", "linear_tape", "double_tape", "helical", "open_arc", "cage_tetrahedral", "cage_octahedral", "bilayer_sheet", "hexagonal_mesh", "nanotube", "multi_helical"], default="floating", help="Topology constraint during diffusion")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Strength of the shape guidance")
    parser.add_argument(
        "--strand_spacing",
        "--target_pitch",
        dest="strand_spacing",
        type=float,
        default=4.8,
        help="Target spacing between adjacent strands for tapes (default: 4.8 A)",
    )
    parser.add_argument("--layer_dist", type=float, default=10.0, help="Target distance between the two layers in double_tape (A)")
    parser.add_argument(
        "--layer_offset_z",
        type=float,
        default=None,
        help="Target longitudinal offset between the two layers in double_tape (A); defaults to 0.5 * strand_spacing",
    )
    parser.add_argument("--target_twist", type=float, default=0.0, help="Target axial twist per step for double_tape / linear_tape (degrees)")
    parser.add_argument("--twist_noise", type=float, default=0.0, help="Standard deviation of noise to add to the axial twist (degrees)")
    parser.add_argument("--target_radius", type=float, default=15.0, help="Target radius for cyclic/cylindrical/helical (A)")
    parser.add_argument("--target_dz", type=float, default=None, help="Target axial translation per chain for helical / nanotube tier pitch / cylindrical intra-unit stagger (A)")
    parser.add_argument("--target_angle", type=float, default=None, help="Target rotation angle per chain for helical / cylindrical (degrees)")
    parser.add_argument("--radius_noise", type=float, default=None, help="Standard deviation of noise to add to the cylinder radius (A); defaults to spacing_noise")
    parser.add_argument("--slide_noise", type=float, default=None, help="Standard deviation of noise to add to the intra-unit helix slide vector (A); defaults to spacing_noise")
    parser.add_argument("--arc_radius", type=float, default=100.0, help="Target radius of curvature for open_arc (A)")
    parser.add_argument("--target_arc_spacing", type=float, default=10.0, help="Target spacing between adjacent chains along the arc (A)")
    parser.add_argument("--target_cage_radius", type=float, default=None, help="Target cage radius (A); defaults to 20.0 for tetrahedral and 35.0 for octahedral cages")
    parser.add_argument("--grid_dim_x", type=int, default=2, help="Grid width for bilayer_sheet / hexagonal_mesh")
    parser.add_argument("--grid_dim_y", type=int, default=2, help="Grid height for bilayer_sheet / hexagonal_mesh")
    parser.add_argument("--row_pitch", type=float, default=10.0, help="In-plane row spacing for bilayer_sheet (A)")
    parser.add_argument("--pore_diameter", type=float, default=None, help="Target pore diameter for hexagonal_mesh (A)")
    parser.add_argument("--lattice_constant", type=float, default=None, help="Hexagonal lattice constant (A), alternative to --pore_diameter")
    parser.add_argument("--ring_size", type=int, default=4, help="K-mer ring size for nanotube")
    parser.add_argument("--num_tiers", type=int, default=4, help="Number of stacked nanotube tiers")
    parser.add_argument("--chiral_stagger", type=float, default=0.0, help="Per-tier nanotube rotation offset (degrees)")
    parser.add_argument("--num_starts", type=int, default=3, help="Number of strands for multi_helical")
    parser.add_argument("--spacing_noise", type=float, default=0.0, help="Standard deviation of noise to add to the spacing target (A)")
    parser.add_argument("--antiparallel_prob", type=float, default=0.0, help="Probability (0.0-1.0) of generating an antiparallel arrangement")
    parser.add_argument("--secondary_structure", type=str, default=None, help="Secondary structure constraint (H, S, L, or a full string)")
    parser.add_argument("--inverse_temp", type=float, default=0.1, help="Sampling temperature for inverse folding (higher = more diverse)")
    parser.add_argument("--seqs_per_backbone", type=int, default=1, help="Number of sequences to generate per structural backbone")
    parser.add_argument("--avoid_aa", type=str, default="", help="String of amino acids to completely avoid (e.g. 'CWP')")
    parser.add_argument("--aa_bias", type=str, default="", help="Amino acid bias string for ProteinMPNN (e.g. 'T:1.0,S:0.5,V:-0.5,I:-0.5')")
    parser.add_argument("--ligand_inclusion_bias", type=float, default=0.0, help="Strength of the inward centripetal force for ligands during diffusion (default: 0.0)")
    parser.add_argument("--ligand_anchor_residue", type=str, default="random", help="Residue index (1-indexed), 'random' to anchor ligand to parent protein, or 'none' to disable (default: random)")
    parser.add_argument("--ligand_anchor_distance", type=float, default=4.0, help="Distance in Angstroms from C_alpha for anchored ligand (default: 4.0)")
    parser.add_argument("--ligand_anchor_direction", type=str, choices=["outward", "radial", "sidechain", "auto"], default="outward", help="Direction to project anchored ligand from C_alpha (default: outward)")
    parser.add_argument("--run", action="store_true", help="Execute BoltzGen after generating YAML")
    parser.add_argument(
        "--heteromer_counter_screen",
        "--heteromer-counter-screen",
        action="store_true",
        help="Design a two-chain heteromer with alternating partner identities and run homomer counter-screens",
    )
    parser.add_argument(
        "--heteromer_charge_bias",
        "--heteromer-charge-bias",
        type=float,
        default=0.5,
        help="Strength of global charge complementarity bias applied to sequences during inverse folding (0 to disable)",
    )
    parser.add_argument(
        "--max_partner_identity",
        "--max-partner-identity",
        type=float,
        default=0.5,
        help="Maximum allowed sequence identity between partners A and B before counter-screening (default: 0.5)",
    )
    parser.add_argument(
        "--max_surviving_designs",
        "--max-surviving-designs",
        type=int,
        default=None,
        help="Maximum number of designs to keep after identity filtering to prevent expensive downstream counter-screening",
    )
    parser.add_argument(
        "--anti_correlation_strength",
        "--anti-correlation-strength",
        type=float,
        default=0.0,
        help="Strength of cross-partner anti-correlation applied during inverse folding (raw logit scale, default: 0.0, recommended: 20.0-50.0 when active)",
    )
    parser.add_argument(
        "--anti_correlation_mode",
        "--anti-correlation-mode",
        type=str,
        default="exact",
        choices=["exact", "blosum62"],
        help="Mode for cross-partner anti-correlation penalty ('exact' or 'blosum62')",
    )
    parser.add_argument(
        "--anti_correlation_surface_only",
        "--anti-correlation-surface-only",
        action="store_true",
        default=False,
        help="Apply cross-partner anti-correlation only to surface (outward-facing) residues, leaving core residues unpenalized",
    )
    parser.add_argument(
        "--partner_orientation",
        "--partner-orientation",
        type=str,
        default="auto",
        choices=["auto", "parallel", "antiparallel"],
        help="Partner strand orientation for identity filtering ('auto', 'parallel', 'antiparallel')",
    )
    parser.add_argument(
        "--identity_filter_weight_core",
        "--identity-filter-weight-core",
        type=float,
        default=0.0,
        help="Weight for core residue identity in zoned partner identity filtering (default: 0.0)",
    )
    parser.add_argument(
        "--identity_filter_weight_surface",
        "--identity-filter-weight-surface",
        type=float,
        default=1.0,
        help="Weight for surface residue identity in zoned partner identity filtering (default: 1.0)",
    )
    parser.add_argument(
        "--identity_filter_zoned",
        "--identity-filter-zoned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable or disable zoned identity filtering (default: auto for tape/sheet topologies)",
    )
    parser.add_argument(
        "--target_max_rmsd",
        type=float,
        default=None,
        help="Optional maximum local RMSD for target models before counter-screening (default: use the filtering RMSD threshold)",
    )
    parser.add_argument(
        "--refolding_rmsd_threshold",
        "--refolding-rmsd-threshold",
        type=float,
        default=None,
        help="RMSD threshold for filtering (default: 2.0 for monomers, recommended 4.0-5.0 for materials)",
    )
    parser.add_argument(
        "--target_min_plddt",
        type=float,
        default=0.0,
        help="Minimum allowed average target pLDDT (0-1 scale) for target models before discarding (default: 0.0)",
    )
    parser.add_argument(
        "--double_tape_core_bias",
        "--double-tape-core-bias",
        type=float,
        default=0.0,
        help="Strength of penalty for T and boost for A in the core (inward-facing) of double-tape (default: 0.0)",
    )
    parser.add_argument(
        "--double_tape_aromatic_bias",
        "--double-tape-aromatic-bias",
        type=float,
        default=1.0,
        help="Strength of boost for F, W and Y on the outside (outward-facing) of double-tape (default: 1.0)",
    )
    parser.add_argument(
        "--double_tape_polar_bias",
        type=float,
        default=0.0,
        help="Strength of boost for R, K, D, E, N, Q, H, T, S on the outside (outward-facing) of double-tape (default: 0.0)",
    )
    parser.add_argument(
        "--filter_composition",
        "--enable_composition_filters",
        action="store_true",
        default=False,
        help="Apply composition and solubility filters for double_tape (outward_aromatic_fraction, design_hydrophobicity, THR_fraction)",
    )
    parser.add_argument(
        "--enable_consecutive_hydrophobics",
        action="store_true",
        default=False,
        help="Enable max_consecutive_ilv_sheet <= 2 filtering for sheet/tape materials",
    )
    parser.add_argument(
        "--backbone_noise",
        type=float,
        default=0.0,
        help="Std (A) of Gaussian noise added to all backbone atoms during inverse folding (default: 0.0; try 0.1-0.2)",
    )
    parser.add_argument(
        "--background_unbias_strength",
        type=float,
        default=0.0,
        help="Strength of natural amino acid background prior removal during inverse folding (default: 0.0; try 0.5-1.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold for inverse folding (default: 1.0 = disabled)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=0.0,
        help="Per-occurrence logit penalty for repeated amino acids within a chain (default: 0.0)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        help="Specific pipeline steps to run (e.g. '--steps design' to generate backbone topologies without the full design pipeline)",
    )
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        action_map = {action.dest: action for action in parser._actions}
        for k, v in config.items():
            norm_k = k.replace("-", "_")
            if norm_k in {"exclude_aa", "exclude_residues"}:
                setattr(args, "avoid_aa", v)
            elif hasattr(args, norm_k) and norm_k != "asym_unit":
                action = action_map.get(norm_k)
                if norm_k == "steps" and isinstance(v, str):
                    v = [v]
                elif action and action.type is not None and v is not None:
                    try:
                        v = action.type(v)
                    except (ValueError, TypeError) as e:
                        parser.error(f"Invalid value for config option '{k}': {e}")
                setattr(args, norm_k, v)

    if "asym_unit" in config:
        asym_unit_def = config["asym_unit"]
    else:
        asym_unit_def = [{"type": "protein", "length": args.length, "secondary_structure": args.secondary_structure} for _ in range(args.asym_unit_size)]

    if args.heteromer_counter_screen and "asym_unit" not in config:
        parser.error(
            "--heteromer_counter_screen requires asym_unit with two distinct, fixed sequence entries"
        )

    protein_templates = [
        item for item in asym_unit_def if str(item.get("type", "protein")).lower() == "protein"
    ]
    args.asym_unit_size = len(protein_templates)
    if not protein_templates:
        parser.error("asym_unit must contain at least one protein template")

    supported_heteromer_topologies = {
        "cyclic",
        "cylindrical",
        "linear_tape",
        "double_tape",
        "open_arc",
        "helical",
    }
    if args.heteromer_counter_screen:
        if len(protein_templates) != 2:
            parser.error(
                "--heteromer_counter_screen requires exactly two protein templates"
            )
        if any(
            str(item.get("type", "protein")).lower() == "ligand"
            for item in asym_unit_def
        ):
            parser.error(
                "--heteromer_counter_screen does not support ligands in the asymmetric unit; "
                "ligand-mediated assembly is a ternary system and is unsuitable for the "
                "current protein-only homomer controls"
            )
        if args.topology not in supported_heteromer_topologies:
            parser.error(
                "--heteromer_counter_screen supports cyclic, cylindrical, linear_tape, "
                "double_tape, open_arc, and helical topologies"
            )
    is_heteromer_assembly = args.heteromer_counter_screen or (
        len(protein_templates) == 2 and args.topology in supported_heteromer_topologies
    )
    if is_heteromer_assembly:
        if args.topology == "double_tape":
            if args.copies % 4 != 0:
                parser.error(
                    f"double_tape with two protein templates requires --copies to be a multiple of 4; got {args.copies}"
                )
        elif args.copies % 2 != 0:
            parser.error(
                f"{args.topology} with two protein templates requires --copies to be an even number (multiple of 2); got {args.copies}"
            )
    else:
        if args.asym_unit_size > 1 and args.copies % args.asym_unit_size != 0:
            parser.error(
                f"--copies ({args.copies}) must be a multiple of the asymmetric unit protein count ({args.asym_unit_size})"
            )
        if args.topology == "double_tape" and args.copies % 2 != 0:
            parser.error(
                f"double_tape requires --copies to be an even number (multiple of 2); got {args.copies}"
            )

    if args.grid_dim_x < 1 or args.grid_dim_y < 1:
        parser.error("--grid_dim_x and --grid_dim_y must be positive")

    if args.target_dz is None:
        if args.topology == "nanotube":
            args.target_dz = 4.8
        elif args.topology == "cylindrical":
            args.target_dz = 0.0
        else:
            args.target_dz = 5.0

    if args.target_angle is None:
        if args.topology == "cylindrical":
            args.target_angle = 0.0
        else:
            args.target_angle = 30.0

    if args.ring_size < 1 or args.num_tiers < 1 or args.num_starts < 1:
        parser.error("--ring_size, --num_tiers, and --num_starts must be positive")

    if args.topology == "bilayer_sheet":
        if args.asym_unit_size != 1:
            parser.error("bilayer_sheet requires exactly one protein template per placement")
        expected_copies = 2 * args.grid_dim_x * args.grid_dim_y
        if args.copies != expected_copies:
            parser.error(
                f"bilayer_sheet requires --copies {expected_copies}; got {args.copies}"
            )
    if args.topology == "hexagonal_mesh":
        if args.asym_unit_size != 1:
            parser.error("hexagonal_mesh requires exactly one protein template per placement")
        expected_copies = 3 * args.grid_dim_x * args.grid_dim_y
        if args.copies != expected_copies:
            parser.error(
                f"hexagonal_mesh requires --copies {expected_copies}; got {args.copies}"
            )
        if args.pore_diameter is None and args.lattice_constant is None:
            parser.error("hexagonal_mesh requires --pore_diameter or --lattice_constant")
        if args.pore_diameter is not None and args.lattice_constant is not None:
            parser.error("specify only one of --pore_diameter or --lattice_constant")
        if args.lattice_constant is None and args.pore_diameter is not None:
            args.lattice_constant = args.pore_diameter + 10.0

    if args.topology == "nanotube":
        if args.asym_unit_size != 1:
            parser.error("nanotube requires exactly one protein template per placement")
        expected_copies = args.ring_size * args.num_tiers
        if args.copies != expected_copies:
            parser.error(
                f"nanotube requires --copies {expected_copies}; got {args.copies}"
            )
    if args.topology == "multi_helical":
        if args.asym_unit_size != 1:
            parser.error("multi_helical requires exactly one protein template per placement")
        if args.copies % args.num_starts != 0:
            parser.error(
                "multi_helical requires --copies divisible by --num_starts "
                f"({args.num_starts})"
            )

    if args.target_cage_radius is None:
        args.target_cage_radius = 20.0 if args.topology == "cage_tetrahedral" else 35.0

    expected_copies = {
        "cage_tetrahedral": 12,
        "cage_octahedral": 24,
    }.get(args.topology)
    if expected_copies is not None and args.copies != expected_copies:
        parser.error(
            f"{args.topology} requires --copies {expected_copies}; got {args.copies}"
        )
    if expected_copies is not None and args.asym_unit_size != 1:
        parser.error(f"{args.topology} requires exactly one protein template per placement")
    
    if args.output_dir == "material_out":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.output_dir}_{timestamp}"
        
    # Set environment variables for the modified diffusion loop
    os.environ["MAT_TOPOLOGY"] = args.topology
    os.environ["MAT_COPIES"] = str(args.copies)
    os.environ["MAT_GUIDANCE_SCALE"] = str(args.guidance_scale)
    os.environ["MAT_LIGAND_INCLUSION_BIAS"] = str(args.ligand_inclusion_bias)
    if args.ligand_anchor_residue is not None:
        os.environ["MAT_LIGAND_ANCHOR_RESIDUE"] = str(args.ligand_anchor_residue)
    else:
        os.environ["MAT_LIGAND_ANCHOR_RESIDUE"] = "random"
    os.environ["MAT_LIGAND_ANCHOR_DISTANCE"] = str(args.ligand_anchor_distance)
    os.environ["MAT_LIGAND_ANCHOR_DIRECTION"] = str(args.ligand_anchor_direction)
    os.environ["MAT_ASYM_UNIT_SIZE"] = str(
        1 if is_heteromer_assembly else args.asym_unit_size
    )
    args.target_pitch = getattr(args, "strand_spacing", getattr(args, "target_pitch", 4.8))
    os.environ["MAT_HETEROMER_SCREEN"] = str(args.heteromer_counter_screen).lower()
    os.environ["MAT_TARGET_PITCH"] = str(args.target_pitch)
    os.environ["MAT_STRAND_SPACING"] = str(args.target_pitch)
    os.environ["MAT_TARGET_RADIUS"] = str(args.target_radius)
    os.environ["MAT_TARGET_DZ"] = str(args.target_dz)
    os.environ["MAT_TARGET_ANGLE"] = str(args.target_angle)
    os.environ["MAT_ARC_RADIUS"] = str(args.arc_radius)
    os.environ["MAT_TARGET_ARC_SPACING"] = str(args.target_arc_spacing)
    os.environ["MAT_SPACING_NOISE"] = str(args.spacing_noise)
    if args.radius_noise is not None:
        os.environ["MAT_RADIUS_NOISE"] = str(args.radius_noise)
    else:
        os.environ.pop("MAT_RADIUS_NOISE", None)
    if args.slide_noise is not None:
        os.environ["MAT_SLIDE_NOISE"] = str(args.slide_noise)
    else:
        os.environ.pop("MAT_SLIDE_NOISE", None)
    os.environ["MAT_ANTIPARALLEL_PROB"] = str(args.antiparallel_prob)
    os.environ["MAT_LAYER_DIST"] = str(args.layer_dist)
    if args.layer_offset_z is not None:
        os.environ["MAT_LAYER_OFFSET_Z"] = str(args.layer_offset_z)
    else:
        os.environ.pop("MAT_LAYER_OFFSET_Z", None)
    os.environ["MAT_TARGET_TWIST"] = str(args.target_twist)
    os.environ["MAT_TWIST_NOISE"] = str(args.twist_noise)
    os.environ["MAT_TARGET_CAGE_RADIUS"] = str(args.target_cage_radius)
    os.environ["MAT_GRID_DIM_X"] = str(args.grid_dim_x)
    os.environ["MAT_GRID_DIM_Y"] = str(args.grid_dim_y)
    os.environ["MAT_ROW_PITCH"] = str(args.row_pitch)
    if args.pore_diameter is not None:
        os.environ["MAT_PORE_DIAMETER"] = str(args.pore_diameter)
    if args.lattice_constant is not None:
        os.environ["MAT_LATTICE_CONSTANT"] = str(args.lattice_constant)
    os.environ["MAT_RING_SIZE"] = str(args.ring_size)
    os.environ["MAT_NUM_TIERS"] = str(args.num_tiers)
    os.environ["MAT_CHIRAL_STAGGER"] = str(args.chiral_stagger)
    os.environ["MAT_NUM_STARTS"] = str(args.num_starts)
    os.environ.pop("MAT_LAYOUT_FILE", None)
    
    yaml_file = generate_yaml_from_spec(
        args.copies,
        asym_unit_def,
        output_file="material_spec.yaml",
        heteromer_counter_screen=args.heteromer_counter_screen,
        topology=args.topology,
        topology_params={
            "target_pitch": args.target_pitch,
            "strand_spacing": args.target_pitch,
            "target_radius": args.target_radius,
            "target_dz": args.target_dz,
            "target_angle": args.target_angle,
            "arc_radius": args.arc_radius,
            "target_arc_spacing": args.target_arc_spacing,
            "layer_dist": args.layer_dist,
            "layer_offset_z": args.layer_offset_z,
            "radius_noise": args.radius_noise,
            "slide_noise": args.slide_noise,
            "target_twist": args.target_twist,
            "twist_noise": args.twist_noise,
            "antiparallel_prob": args.antiparallel_prob,
        },
        charge_bias_strength=args.heteromer_charge_bias,
    )
    
    if args.run:
        asym_size = 1 if is_heteromer_assembly else args.asym_unit_size
        n_units = args.copies if is_heteromer_assembly else args.copies // args.asym_unit_size
        if "asym_unit" in config:
            print(f"Running BoltzGen with {args.copies} total chains ({n_units} asymmetric units of size {asym_size})...")
        else:
            print(f"Running BoltzGen with {args.copies} total chains of length {args.length}...")
        print(f"Topology constraint: {args.topology} (Asym Unit Size: {asym_size})")
        
        cmd = [
            "boltzgen", "run", yaml_file,
            "--output", args.output_dir,
            "--protocol", "peptide-anything",
            "--num_designs", str(args.num_designs),
            "--diffusion_batch_size", str(args.diffusion_batch_size),
            "--inverse_fold_num_sequences", str(args.seqs_per_backbone),
            "--config", "inverse_folding", f"override.inverse_fold_args.sampling_temperature={args.inverse_temp}"
        ]
        if args.heteromer_counter_screen:
            cmd.append("--heteromer_counter_screen")
            if hasattr(args, "max_partner_identity"):
                cmd.extend(["--max_partner_identity", str(args.max_partner_identity)])
            if hasattr(args, "max_surviving_designs") and args.max_surviving_designs is not None:
                cmd.extend(["--max_surviving_designs", str(args.max_surviving_designs)])
            if args.target_max_rmsd is not None:
                cmd.extend(["--target_max_rmsd", str(args.target_max_rmsd)])
            elif getattr(args, "refolding_rmsd_threshold", None) is not None:
                cmd.extend(["--target_max_rmsd", str(args.refolding_rmsd_threshold)])
            if hasattr(args, "target_min_plddt"):
                cmd.extend(["--target_min_plddt", str(args.target_min_plddt)])

        if getattr(args, "refolding_rmsd_threshold", None) is not None:
            cmd.extend(["--refolding_rmsd_threshold", str(args.refolding_rmsd_threshold)])

        if getattr(args, "anti_correlation_strength", 0.0) > 0.0:
            cmd.extend(["--anti_correlation_strength", str(args.anti_correlation_strength)])
            if getattr(args, "anti_correlation_mode", "exact") != "exact":
                cmd.extend(["--anti_correlation_mode", str(args.anti_correlation_mode)])

        if getattr(args, "anti_correlation_surface_only", False):
            cmd.append("--anti_correlation_surface_only")
        if getattr(args, "partner_orientation", "auto") != "auto":
            cmd.extend(["--partner_orientation", str(args.partner_orientation)])
        if getattr(args, "identity_filter_weight_core", 0.0) != 0.0:
            cmd.extend(["--identity_filter_weight_core", str(args.identity_filter_weight_core)])
        if getattr(args, "identity_filter_weight_surface", 1.0) != 1.0:
            cmd.extend(["--identity_filter_weight_surface", str(args.identity_filter_weight_surface)])
        if getattr(args, "identity_filter_zoned", None) is not None:
            cmd.append(f"--{'identity-filter-zoned' if args.identity_filter_zoned else 'no-identity-filter-zoned'}")

        if getattr(args, "double_tape_core_bias", 0.0) > 0.0:
            cmd.extend(["--double_tape_core_bias", str(args.double_tape_core_bias)])
        if getattr(args, "double_tape_aromatic_bias", 0.0) > 0.0:
            cmd.extend(["--double_tape_aromatic_bias", str(args.double_tape_aromatic_bias)])
        if getattr(args, "double_tape_polar_bias", 0.0) > 0.0:
            cmd.extend(["--double_tape_polar_bias", str(args.double_tape_polar_bias)])
        if getattr(args, "backbone_noise", 0.0) > 0.0:
            cmd.extend(["--backbone_noise", str(args.backbone_noise)])
        if getattr(args, "background_unbias_strength", 0.0) > 0.0:
            cmd.extend(["--background_unbias_strength", str(args.background_unbias_strength)])
        if getattr(args, "top_p", 1.0) < 1.0:
            cmd.extend(["--top_p", str(args.top_p)])
        if getattr(args, "repetition_penalty", 0.0) > 0.0:
            cmd.extend(["--repetition_penalty", str(args.repetition_penalty)])

        material_filtering_override = format_metrics_override(args.topology)
        if material_filtering_override is not None:
            cmd.extend(["--config", "filtering", material_filtering_override])
            
        additional_filters_override = format_additional_filters(
            args.topology,
            enable_consecutive_hydrophobics=getattr(args, "enable_consecutive_hydrophobics", False),
            is_heteromer=is_heteromer_assembly,
            enable_composition_filters=getattr(args, "filter_composition", False),
        )
        if additional_filters_override is not None:
            cmd.extend(["--config", "filtering", additional_filters_override])
        
        if args.avoid_aa:
            cmd.extend(["--inverse_fold_avoid", args.avoid_aa])
        if args.aa_bias:
            cmd.append(f"override.inverse_fold_args.aa_bias={args.aa_bias}")
        if getattr(args, "steps", None):
            steps_list = [args.steps] if isinstance(args.steps, str) else list(args.steps)
            cmd.extend(["--steps"] + steps_list)
        
        try:
            subprocess.run(cmd, check=True)
            print("BoltzGen run completed successfully.")
            
            # Find the output structures to calculate BSA
            # Check filtering output first, fallback to refold, then to intermediate designs
            output_cifs = glob.glob(os.path.join(args.output_dir, "filtering", "*.cif"))
            if not output_cifs:
                output_cifs = glob.glob(os.path.join(args.output_dir, "intermediate_designs_inverse_folded", "refold_cif", "*.cif"))
            if not output_cifs:
                output_cifs = glob.glob(os.path.join(args.output_dir, "intermediate_designs", "*.cif"))
            
            import pandas as pd
            metrics_csvs = glob.glob(os.path.join(args.output_dir, "final_ranked_designs", "metrics_*.csv"))
            if metrics_csvs:
                df = pd.read_csv(metrics_csvs[0])
                df = df.sort_values("max_rank")
                print("\n--- TOP RESULTS BY COMPOSITE METRICS ---")
                cols_to_print = ["id", "max_rank", "neg_lattice_rmsd_refolded", "h_bonds_per_interface_refolded", "packing_density_refolded", "design_to_target_iptm"]
                available_cols = [c for c in cols_to_print if c in df.columns]
                
                # Format to nice strings
                print(df[available_cols].head(10).to_string(index=False))
            elif output_cifs:
                print(f"Generated {len(output_cifs)} design structure(s): {', '.join([os.path.basename(c) for c in output_cifs[:5]])}{'...' if len(output_cifs) > 5 else ''}")
            else:
                print("No output metrics CSV found.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running BoltzGen: {e}")

if __name__ == "__main__":
    main()
