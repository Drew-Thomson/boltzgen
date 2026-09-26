import argparse
import os
import subprocess
import yaml
import glob
import datetime
from pathlib import Path
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

def generate_yaml_from_spec(num_copies, asym_unit_def, output_file="material_spec.yaml"):
    entities = []
    asym_unit_size = len(asym_unit_def)
    total_chains = num_copies * asym_unit_size
    
    chain_ids = [get_chain_id(i) for i in range(total_chains)]
    
    for copy_idx in range(num_copies):
        for chain_idx_in_unit, chain_def in enumerate(asym_unit_def):
            global_chain_idx = copy_idx * asym_unit_size + chain_idx_in_unit
            c_id = chain_ids[global_chain_idx]
            
            ent_type = chain_def.get("type", "protein")
            ent_dict = {"id": c_id}
            
            if ent_type == "protein":
                ent_dict["sequence"] = str(chain_def.get("length", 15))
                ent_dict["symmetric_group"] = chain_def.get("symmetric_group", chain_idx_in_unit + 1)
                
                sec_struct = chain_def.get("secondary_structure")
                if sec_struct:
                    length = chain_def.get("length", 15)
                    if len(sec_struct) == 1:
                        ent_dict["secondary_structure"] = sec_struct * length
                    else:
                        ent_dict["secondary_structure"] = sec_struct
            elif ent_type == "ligand":
                if "ccd" in chain_def:
                    ent_dict["ccd"] = chain_def["ccd"]
                if "smiles" in chain_def:
                    ent_dict["smiles"] = chain_def["smiles"]
            
            # Carry over any other keys natively to BoltzGen (like residue_constraints)
            for k, v in chain_def.items():
                if k not in ["type", "length", "secondary_structure", "symmetric_group", "ccd", "smiles"]:
                    ent_dict[k] = v
                    
            entities.append({ent_type: ent_dict})
            
    spec = {"entities": entities}
    with open(output_file, "w") as f:
        yaml.dump(spec, f, sort_keys=False)
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
    parser.add_argument("--asym_unit_size", type=int, default=1, help="Number of chains in the asymmetric unit")
    parser.add_argument("--topology", type=str, choices=["floating", "cyclic", "linear_tape", "double_tape", "helical", "open_arc"], default="floating", help="Topology constraint during diffusion")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Strength of the shape guidance")
    parser.add_argument("--target_pitch", type=float, default=10.0, help="Target spacing between adjacent chains for tapes (A)")
    parser.add_argument("--layer_dist", type=float, default=10.0, help="Target distance between the two layers in double_tape (A)")
    parser.add_argument("--target_radius", type=float, default=15.0, help="Target radius for cyclic/helical (A)")
    parser.add_argument("--target_dz", type=float, default=5.0, help="Target axial translation per chain for helical (A)")
    parser.add_argument("--target_angle", type=float, default=30.0, help="Target rotation angle per chain for helical (degrees)")
    parser.add_argument("--arc_radius", type=float, default=100.0, help="Target radius of curvature for open_arc (A)")
    parser.add_argument("--target_arc_spacing", type=float, default=10.0, help="Target spacing between adjacent chains along the arc (A)")
    parser.add_argument("--spacing_noise", type=float, default=0.0, help="Standard deviation of noise to add to the spacing target (A)")
    parser.add_argument("--antiparallel_prob", type=float, default=0.0, help="Probability (0.0-1.0) of generating an antiparallel arrangement")
    parser.add_argument("--secondary_structure", type=str, default=None, help="Secondary structure constraint (H, S, L, or a full string)")
    parser.add_argument("--inverse_temp", type=float, default=0.1, help="Sampling temperature for inverse folding (higher = more diverse)")
    parser.add_argument("--seqs_per_backbone", type=int, default=1, help="Number of sequences to generate per structural backbone")
    parser.add_argument("--avoid_aa", type=str, default="", help="String of amino acids to completely avoid (e.g. 'CWP')")
    parser.add_argument("--run", action="store_true", help="Execute BoltzGen after generating YAML")
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        for k, v in config.items():
            if hasattr(args, k) and k != "asym_unit":
                setattr(args, k, v)
                
    if "asym_unit" in config:
        asym_unit_def = config["asym_unit"]
        args.asym_unit_size = len(asym_unit_def)
    else:
        asym_unit_def = [{"type": "protein", "length": args.length, "secondary_structure": args.secondary_structure} for _ in range(args.asym_unit_size)]
    
    if args.output_dir == "material_out":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.output_dir}_{timestamp}"
        
    # Set environment variables for the modified diffusion loop
    os.environ["MAT_TOPOLOGY"] = args.topology
    os.environ["MAT_GUIDANCE_SCALE"] = str(args.guidance_scale)
    os.environ["MAT_ASYM_UNIT_SIZE"] = str(args.asym_unit_size)
    os.environ["MAT_TARGET_PITCH"] = str(args.target_pitch)
    os.environ["MAT_TARGET_RADIUS"] = str(args.target_radius)
    os.environ["MAT_TARGET_DZ"] = str(args.target_dz)
    os.environ["MAT_TARGET_ANGLE"] = str(args.target_angle)
    os.environ["MAT_ARC_RADIUS"] = str(args.arc_radius)
    os.environ["MAT_TARGET_ARC_SPACING"] = str(args.target_arc_spacing)
    os.environ["MAT_SPACING_NOISE"] = str(args.spacing_noise)
    os.environ["MAT_ANTIPARALLEL_PROB"] = str(args.antiparallel_prob)
    os.environ["MAT_LAYER_DIST"] = str(args.layer_dist)
    
    yaml_file = generate_yaml_from_spec(args.copies, asym_unit_def, output_file="material_spec.yaml")
    
    if args.run:
        if "asym_unit" in config:
            print(f"Running BoltzGen with {args.copies} copies of an asymmetric unit of size {args.asym_unit_size}...")
        else:
            print(f"Running BoltzGen with {args.copies} copies of length {args.length}...")
        print(f"Topology constraint: {args.topology} (Asym Unit Size: {args.asym_unit_size})")
        
        cmd = [
            "boltzgen", "run", yaml_file,
            "--output", args.output_dir,
            "--protocol", "peptide-anything",
            "--num_designs", str(args.num_designs),
            "--inverse_fold_num_sequences", str(args.seqs_per_backbone),
            "--config", "inverse_folding", f"override.inverse_fold_args.sampling_temperature={args.inverse_temp}"
        ]
        
        if args.avoid_aa:
            cmd.extend(["--inverse_fold_avoid", args.avoid_aa])
        
        try:
            subprocess.run(cmd, check=True)
            print("BoltzGen run completed successfully.")
            
            # Find the output structures to calculate BSA
            # Check filtering output first, fallback to refold
            output_cifs = glob.glob(os.path.join(args.output_dir, "filtering", "*.cif"))
            if not output_cifs:
                output_cifs = glob.glob(os.path.join(args.output_dir, "intermediate_designs_inverse_folded", "refold_cif", "*.cif"))
            
            if output_cifs:
                results = []
                print(f"\nAnalyzing {len(output_cifs)} generated structures...")
                for struct_file in output_cifs:
                    bsa, sasa = calculate_bsa(struct_file)
                    ratio = bsa/sasa if sasa > 0 else 0
                    results.append({"file": os.path.basename(struct_file), "bsa": bsa, "sasa": sasa, "ratio": ratio})
                
                # Sort by BSA descending (highest BSA first)
                results.sort(key=lambda x: x["bsa"], reverse=True)
                
                print("\n--- RESULTS RANKED BY BURIED SURFACE AREA ---")
                print(f"{'Filename':<35} | {'BSA (A^2)':<10} | {'SASA (A^2)':<10} | {'BSA/SASA'}")
                print("-" * 75)
                for res in results:
                    print(f"{res['file']:<35} | {res['bsa']:<10.2f} | {res['sasa']:<10.2f} | {res['ratio']:.3f}")
            else:
                print("No output CIF files found to calculate BSA.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running BoltzGen: {e}")

if __name__ == "__main__":
    main()
