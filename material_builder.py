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

def generate_yaml(num_chains, length, output_file="material_spec.yaml"):
    entities = []
    # Letters for chains A, B, C, D...
    chain_ids = [chr(65 + i) for i in range(num_chains)]
    
    for c_id in chain_ids:
        entities.append({
            "protein": {
                "id": c_id,
                "sequence": str(length),
                "symmetric_group": 1
            }
        })
        
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
    parser.add_argument("--copies", type=int, default=4, help="Number of identical peptide chains")
    parser.add_argument("--length", type=int, default=15, help="Length of the peptide")
    parser.add_argument("--output_dir", type=str, default="material_out", help="Directory for BoltzGen outputs")
    parser.add_argument("--num_designs", type=int, default=1, help="Number of design candidates to generate")
    parser.add_argument("--topology", type=str, choices=["floating", "cyclic", "linear_tape", "helical", "open_arc"], default="floating", help="Topology constraint during diffusion")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Strength of the shape guidance")
    parser.add_argument("--asym_unit_size", type=int, default=1, help="Number of chains forming a single asymmetric repeating unit (e.g. 2 for a two-layer fibre)")
    parser.add_argument("--target_pitch", type=float, default=10.0, help="Target spacing between adjacent chains for linear_tape (A)")
    parser.add_argument("--target_radius", type=float, default=30.0, help="Target radius for cyclic (A)")
    parser.add_argument("--target_dz", type=float, default=5.0, help="Target axial translation per chain for helical (A)")
    parser.add_argument("--target_angle", type=float, default=30.0, help="Target rotation angle per chain for helical (degrees)")
    parser.add_argument("--arc_radius", type=float, default=100.0, help="Target radius of curvature for open_arc (A)")
    parser.add_argument("--target_arc_spacing", type=float, default=10.0, help="Target spacing between adjacent chains along the arc (A)")
    parser.add_argument("--run", action="store_true", help="Execute BoltzGen after generating YAML")
    
    args = parser.parse_args()
    
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
    
    yaml_file = generate_yaml(args.copies, args.length)
    
    if args.run:
        print(f"Running BoltzGen with {args.copies} copies of length {args.length}...")
        print(f"Topology constraint: {args.topology} (Asym Unit Size: {args.asym_unit_size})")
        
        cmd = [
            "boltzgen", "run", yaml_file,
            "--output", args.output_dir,
            "--num_designs", str(args.num_designs)
        ]
        
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
