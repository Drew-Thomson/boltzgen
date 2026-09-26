with open("src/boltzgen/model/modules/diffusion.py", "r") as f:
    content = f.read()

import re

# We will replace the target logic for linear_tape, helical, cyclic, open_arc

# For linear_tape:
# old:
#                             if topology == "linear_tape":
#                                 target_pitch = float(os.environ.get("MAT_TARGET_PITCH", "10.0"))
#                                 sorted_proj = projections[sort_idx]
#                                 adaptive_pitch = (sorted_proj[-1] - sorted_proj[0]) / max(1, N_units - 1)
#                                 pitch = target_pitch * blend_alpha + adaptive_pitch * (1 - blend_alpha)
# new:
#                             if topology == "linear_tape":
#                                 pitch = float(os.environ.get("MAT_TARGET_PITCH", "10.0"))

old_linear = """                            if topology == "linear_tape":
                                target_pitch = float(os.environ.get("MAT_TARGET_PITCH", "10.0"))
                                sorted_proj = projections[sort_idx]
                                adaptive_pitch = (sorted_proj[-1] - sorted_proj[0]) / max(1, N_units - 1)
                                pitch = target_pitch * blend_alpha + adaptive_pitch * (1 - blend_alpha)"""

new_linear = """                            if topology == "linear_tape":
                                pitch = float(os.environ.get("MAT_TARGET_PITCH", "10.0"))"""
                                
old_helical = """                            elif topology == "helical":
                                target_dz = float(os.environ.get("MAT_TARGET_DZ", "5.0"))
                                target_angle_deg = float(os.environ.get("MAT_TARGET_ANGLE", "30.0"))
                                target_angle = target_angle_deg * math.pi / 180.0
                                
                                sorted_proj = projections[sort_idx]
                                adaptive_dz = (sorted_proj[-1] - sorted_proj[0]) / max(1, N_units - 1)
                                dz = target_dz * blend_alpha + adaptive_dz * (1 - blend_alpha)
                                angle = target_angle"""
                                
new_helical = """                            elif topology == "helical":
                                dz = float(os.environ.get("MAT_TARGET_DZ", "5.0"))
                                target_angle_deg = float(os.environ.get("MAT_TARGET_ANGLE", "30.0"))
                                angle = target_angle_deg * math.pi / 180.0"""
                                
old_cyclic = """                                    if topology in ["cyclic", "open_arc"]:
                                        target_radius = float(os.environ.get("MAT_TARGET_RADIUS", "30.0"))
                                        target_r = target_radius if topology == "cyclic" else arc_radius
                                        consensus_com = consensus_m.mean(dim=0)
                                        r_vec = consensus_com - torch.dot(consensus_com, axis_z) * axis_z
                                        r_dist = torch.norm(r_vec)
                                        if r_dist > 1e-3:
                                            r_dir = r_vec / r_dist
                                            adaptive_r = r_dist
                                            r_target = target_r * blend_alpha + adaptive_r * (1 - blend_alpha)
                                            consensus_m = consensus_m + (r_target - adaptive_r) * r_dir"""
                                            
new_cyclic = """                                    if topology in ["cyclic", "open_arc"]:
                                        target_radius = float(os.environ.get("MAT_TARGET_RADIUS", "30.0"))
                                        target_r = target_radius if topology == "cyclic" else arc_radius
                                        consensus_com = consensus_m.mean(dim=0)
                                        r_vec = consensus_com - torch.dot(consensus_com, axis_z) * axis_z
                                        r_dist = torch.norm(r_vec)
                                        if r_dist > 1e-3:
                                            r_dir = r_vec / r_dist
                                            consensus_m = consensus_m + (target_r - r_dist) * r_dir"""

# Also boost the gradient scaling:
# old:
#                     if apply_grad:
#                         scale = guidance_scale * current_t * 0.5
# new:
#                     if apply_grad:
#                         scale = guidance_scale * current_t * 1.5

old_scale = """                    if apply_grad:
                        scale = guidance_scale * current_t * 0.5
                        atom_coords_denoised = atom_coords_denoised - scale * grad_tensor"""
                        
new_scale = """                    if apply_grad:
                        scale = guidance_scale * current_t * 1.5
                        atom_coords_denoised = atom_coords_denoised - scale * grad_tensor"""

content = content.replace(old_linear, new_linear)
content = content.replace(old_helical, new_helical)
content = content.replace(old_cyclic, new_cyclic)
content = content.replace(old_scale, new_scale)

with open("src/boltzgen/model/modules/diffusion.py", "w") as f:
    f.write(content)
