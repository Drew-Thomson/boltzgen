with open("src/boltzgen/model/modules/diffusion.py", "r") as f:
    content = f.read()

start_str = """            import os
            import math
            topology = os.environ.get("MAT_TOPOLOGY", "floating")"""

end_str = """            if self.alignment_reverse_diff:"""

start_idx = content.find(start_str)
end_idx = content.find(end_str, start_idx)

if start_idx == -1 or end_idx == -1:
    print("Could not find boundaries")
    import sys
    sys.exit(1)

new_code = """            import os
            import math
            topology = os.environ.get("MAT_TOPOLOGY", "floating")
            if topology in ["linear_tape", "cyclic", "helical", "open_arc"]:
                guidance_scale = float(os.environ.get("MAT_GUIDANCE_SCALE", "1.0"))
                asym_unit_size = int(os.environ.get("MAT_ASYM_UNIT_SIZE", "1"))
                
                feats = network_condition_kwargs["feats"]
                import sys
                is_folding = any("folding" in arg for arg in sys.argv)
                is_designing = "design_mask" in feats and feats["design_mask"].sum() > 0
                
                if not is_folding and is_designing and "asym_id" in feats and "atom_to_token" in feats and guidance_scale > 0:
                    asym_id = feats["asym_id"] 
                    atom_to_token = feats["atom_to_token"]
                    b, m, _ = atom_coords_denoised.shape
                    
                    if asym_id.dim() == 1: asym_id = asym_id.unsqueeze(0)
                    if asym_id.shape[0] == 1 and b > 1: asym_id = asym_id.expand(b, -1)
                        
                    if atom_to_token.dim() == 2: atom_to_token = atom_to_token.unsqueeze(0)
                    if atom_to_token.shape[0] == 1 and b > 1: atom_to_token = atom_to_token.expand(b, -1, -1)
                    
                    token_indices = atom_to_token.long().argmax(dim=-1)
                    
                    grad_tensor = torch.zeros_like(atom_coords_denoised)
                    apply_grad = False
                    
                    current_t = float(t_hat.max().item()) if hasattr(t_hat, "max") else float(t_hat)
                    blend_alpha = min(max(current_t / 1.0, 0.0), 1.0) 
                    
                    for batch_idx in range(b):
                        atom_asym_id = asym_id[batch_idx, token_indices[batch_idx]]
                        chain_ids = torch.unique(atom_asym_id)
                        
                        if len(chain_ids) >= 2 * asym_unit_size:
                            N_chains = len(chain_ids)
                            N_units = N_chains // asym_unit_size
                            
                            if N_units < 2: continue
                            
                            unit_masks = []
                            for u in range(N_units):
                                u_chain_ids = chain_ids[u*asym_unit_size : (u+1)*asym_unit_size]
                                mask_u = torch.zeros_like(atom_asym_id, dtype=torch.bool)
                                for c_id in u_chain_ids:
                                    mask_u |= (atom_asym_id == c_id)
                                mask_u &= atom_mask[batch_idx].bool()
                                unit_masks.append(mask_u)
                                
                            unit_coms = []
                            for mask_u in unit_masks:
                                if mask_u.sum() > 0:
                                    unit_coms.append(atom_coords_denoised[batch_idx, mask_u].mean(dim=0))
                                else:
                                    unit_coms.append(torch.zeros(3, device=atom_coords_denoised.device))
                            unit_coms = torch.stack(unit_coms) 
                            
                            overall_com = unit_coms.mean(dim=0)
                            centered_coms = unit_coms - overall_com
                            
                            cov = centered_coms.T @ centered_coms
                            U, S, Vh = torch.linalg.svd(cov.to(torch.float32))
                            U = U.to(atom_coords_denoised.dtype)
                            
                            axis_z = U[:, 0]
                            
                            if topology in ["linear_tape", "helical"]:
                                projections = (centered_coms @ axis_z.unsqueeze(1)).squeeze(1)
                                sort_idx = torch.argsort(projections)
                                
                                if projections[-1] < projections[0]:
                                    axis_z = -axis_z
                                    projections = (centered_coms @ axis_z.unsqueeze(1)).squeeze(1)
                                    sort_idx = torch.argsort(projections)
                            elif topology in ["cyclic", "open_arc"]:
                                axis_z = U[:, 2] 
                                axis_x = U[:, 0]
                                axis_y = U[:, 1]
                                proj_x = (centered_coms @ axis_x.unsqueeze(1)).squeeze(1)
                                proj_y = (centered_coms @ axis_y.unsqueeze(1)).squeeze(1)
                                angles = torch.atan2(proj_y, proj_x)
                                sort_idx = torch.argsort(angles)
                                
                                seq_normal_sum = torch.zeros_like(axis_z)
                                for j in range(N_units - 1):
                                    seq_normal_sum += torch.linalg.cross(centered_coms[j], centered_coms[j+1])
                                if torch.dot(seq_normal_sum, axis_z) < 0:
                                    axis_z = -axis_z
                                    axis_y = -axis_y
                                    proj_y = (centered_coms @ axis_y.unsqueeze(1)).squeeze(1)
                                    angles = torch.atan2(proj_y, proj_x)
                                    sort_idx = torch.argsort(angles)
                            
                            ranks = torch.empty_like(sort_idx)
                            ranks[sort_idx] = torch.arange(N_units, device=sort_idx.device)
                            
                            if topology == "linear_tape":
                                target_pitch = float(os.environ.get("MAT_TARGET_PITCH", "10.0"))
                                sorted_proj = projections[sort_idx]
                                adaptive_pitch = (sorted_proj[-1] - sorted_proj[0]) / max(1, N_units - 1)
                                pitch = target_pitch * blend_alpha + adaptive_pitch * (1 - blend_alpha)
                                
                            elif topology == "helical":
                                target_dz = float(os.environ.get("MAT_TARGET_DZ", "5.0"))
                                target_angle_deg = float(os.environ.get("MAT_TARGET_ANGLE", "30.0"))
                                target_angle = target_angle_deg * math.pi / 180.0
                                
                                sorted_proj = projections[sort_idx]
                                adaptive_dz = (sorted_proj[-1] - sorted_proj[0]) / max(1, N_units - 1)
                                dz = target_dz * blend_alpha + adaptive_dz * (1 - blend_alpha)
                                angle = target_angle 
                                
                            if topology in ["linear_tape", "helical"]:
                                axis_z = U[:, 0]
                                axis_x = U[:, 1]
                                axis_y = U[:, 2]
                            else:
                                axis_z = U[:, 2]
                                axis_x = U[:, 0]
                                axis_y = U[:, 1]
                            
                            def rotation_matrix_around_axis(axis, theta):
                                axis = axis / torch.norm(axis)
                                a = math.cos(theta)
                                b = math.sin(theta)
                                u, v, w = axis[0], axis[1], axis[2]
                                return torch.tensor([
                                    [a + u*u*(1-a), u*v*(1-a) - w*b, u*w*(1-a) + v*b],
                                    [v*u*(1-a) + w*b, a + v*v*(1-a), v*w*(1-a) - u*b],
                                    [w*u*(1-a) - v*b, w*v*(1-a) + u*b, a + w*w*(1-a)]
                                ], device=axis.device, dtype=axis.dtype)
                                
                            canonical_atoms_list = []
                            valid_unit_masks = []
                            
                            for u in range(N_units):
                                mask_u = unit_masks[u]
                                if mask_u.sum() == 0: continue
                                
                                coords_u = atom_coords_denoised[batch_idx, mask_u] 
                                rank = ranks[u].item() 
                                
                                if topology == "linear_tape":
                                    p_local = (rank - (N_units-1)/2.0) * pitch * axis_z
                                    m_coords = coords_u - overall_com - p_local
                                    
                                elif topology == "helical":
                                    p_local = (rank - (N_units-1)/2.0) * dz * axis_z
                                    theta = (rank - (N_units-1)/2.0) * angle
                                    c_shifted = coords_u - overall_com - p_local
                                    rot_inv = rotation_matrix_around_axis(axis_z, -theta)
                                    m_coords = c_shifted @ rot_inv.T
                                    
                                elif topology == "cyclic":
                                    theta = rank * (2 * math.pi / N_units)
                                    c_shifted = coords_u - overall_com
                                    rot_inv = rotation_matrix_around_axis(axis_z, -theta)
                                    m_coords = c_shifted @ rot_inv.T
                                    
                                elif topology == "open_arc":
                                    arc_radius = float(os.environ.get("MAT_ARC_RADIUS", "100.0"))
                                    target_arc_spacing = float(os.environ.get("MAT_TARGET_ARC_SPACING", "10.0"))
                                    ratio = min(target_arc_spacing / (2.0 * arc_radius + 1e-8), 1.0)
                                    angle_step = 2.0 * math.asin(ratio)
                                    theta = (rank - (N_units-1)/2.0) * angle_step
                                    
                                    c_shifted = coords_u - overall_com
                                    rot_inv = rotation_matrix_around_axis(axis_z, -theta)
                                    m_coords = c_shifted @ rot_inv.T
                                
                                canonical_atoms_list.append(m_coords)
                                valid_unit_masks.append(mask_u)
                                
                            if len(canonical_atoms_list) > 0:
                                min_atoms = min(m.shape[0] for m in canonical_atoms_list)
                                if all(m.shape[0] == min_atoms for m in canonical_atoms_list):
                                    stacked_m = torch.stack(canonical_atoms_list) 
                                    consensus_m = stacked_m.mean(dim=0) 
                                    
                                    if topology in ["cyclic", "open_arc"]:
                                        target_radius = float(os.environ.get("MAT_TARGET_RADIUS", "30.0"))
                                        target_r = target_radius if topology == "cyclic" else arc_radius
                                        consensus_com = consensus_m.mean(dim=0)
                                        r_vec = consensus_com - torch.dot(consensus_com, axis_z) * axis_z
                                        r_dist = torch.norm(r_vec)
                                        if r_dist > 1e-3:
                                            r_dir = r_vec / r_dist
                                            adaptive_r = r_dist
                                            r_target = target_r * blend_alpha + adaptive_r * (1 - blend_alpha)
                                            consensus_m = consensus_m + (r_target - adaptive_r) * r_dir
                                    
                                    for u in range(len(valid_unit_masks)):
                                        rank = ranks[u].item()
                                        mask_u = valid_unit_masks[u]
                                        
                                        if topology == "linear_tape":
                                            p_local = (rank - (N_units-1)/2.0) * pitch * axis_z
                                            c_ideal = consensus_m + overall_com + p_local
                                            
                                        elif topology == "helical":
                                            p_local = (rank - (N_units-1)/2.0) * dz * axis_z
                                            theta = (rank - (N_units-1)/2.0) * angle
                                            rot_fwd = rotation_matrix_around_axis(axis_z, theta)
                                            c_ideal = (consensus_m @ rot_fwd.T) + overall_com + p_local
                                            
                                        elif topology == "cyclic":
                                            theta = rank * (2 * math.pi / N_units)
                                            rot_fwd = rotation_matrix_around_axis(axis_z, theta)
                                            c_ideal = (consensus_m @ rot_fwd.T) + overall_com
                                            
                                        elif topology == "open_arc":
                                            angle_step = 2.0 * math.asin(min(float(os.environ.get("MAT_TARGET_ARC_SPACING", "10.0")) / (2.0 * float(os.environ.get("MAT_ARC_RADIUS", "100.0")) + 1e-8), 1.0))
                                            theta = (rank - (N_units-1)/2.0) * angle_step
                                            rot_fwd = rotation_matrix_around_axis(axis_z, theta)
                                            c_ideal = (consensus_m @ rot_fwd.T) + overall_com
                                            
                                        grad = atom_coords_denoised[batch_idx, mask_u] - c_ideal
                                        grad = torch.clamp(grad, min=-5.0, max=5.0)
                                        grad_tensor[batch_idx, mask_u] += grad
                                        apply_grad = True

                    if apply_grad:
                        scale = guidance_scale * current_t * 0.5
                        atom_coords_denoised = atom_coords_denoised - scale * grad_tensor\n\n"""

with open("src/boltzgen/model/modules/diffusion.py", "w") as f:
    f.write(content[:start_idx] + new_code + content[end_idx:])
