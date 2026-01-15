#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                                                                               
                     
                                                                               
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

COLORS = {
    'solar': '#8B7DD8',
    'glm': '#4B4B4B',
    'angle': '#E07B54',
    'bg': '#FFFFFF',
    'grid': '#CCCCCC',
    'text': '#333333',
    'octant': '#F5F5F5',
}

                                                                               
                           
                                                                               
                                                                    
                                                                          
                                            
S = 0.7
D1, D2 = 0.25, 0.35
V1 = np.array([S + D1, S, S + 0.02])                                       
V2 = np.array([S, S + D2, S + 0.02])                                       

def center_vector(v):
    """Center vector by its own mean (Pearson centering)."""
    return v - np.mean(v)

                                                                               
                   
                                                                               
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(cos_sim(a, b), -1, 1)))

def setup_3d_axes(ax, show_octant=True, lim=1.1):
    """Setup 3D axes with positive octant visualization."""
    ax.set_facecolor(COLORS['bg'])

                     
    if show_octant:
        s = lim * 0.9
        verts_xy = [[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]]
        ax.add_collection3d(Poly3DCollection(
            [verts_xy], alpha=0.15, facecolor='#F0F0F0',
            edgecolor='#CCCCCC', linewidth=0.5
        ))

                                                                   
    axis_len = lim * 0.85
    neg_len = 0.25                                           
    ax.plot([-neg_len, axis_len], [0, 0], [0, 0], color='#888888', lw=1.5, zorder=1)
    ax.plot([0, 0], [-neg_len, axis_len], [0, 0], color='#888888', lw=1.5, zorder=1)
    ax.plot([0, 0], [0, 0], [-neg_len, axis_len], color='#888888', lw=1.5, zorder=1)

            
    ax.scatter([0], [0], [0], color='#666666', s=50, zorder=10)

                                                                   
    margin = 0.3
    ax.set_xlim(-margin, lim)
    ax.set_ylim(-margin, lim)
    ax.set_zlim(-margin, lim)

              
    ax.set_axis_off()

def draw_vector_3d(ax, v, color, label, label_offset=(0.03, 0.03, 0.03)):
    """Draw a 3D vector with arrow and label."""
    ax.quiver(0, 0, 0, v[0], v[1], v[2],
              color=color, arrow_length_ratio=0.12, linewidth=3.5)
    ax.text(v[0] + label_offset[0], v[1] + label_offset[1], v[2] + label_offset[2],
            label, fontsize=14, fontweight='bold', color=color)

def draw_arc_3d(ax, v1, v2, color, alpha=0.3):
    """Draw arc between two vectors in 3D."""
                       
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

                       
    n_points = 30
    arc_radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.3

    points = []
    for t in np.linspace(0, 1, n_points):
                                         
        omega = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))
        if omega < 0.001:
            p = v1_norm
        else:
            p = (np.sin((1-t)*omega) * v1_norm + np.sin(t*omega) * v2_norm) / np.sin(omega)
        points.append(p * arc_radius)

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], points[:, 2],
            color=color, linewidth=2, alpha=0.8)

def draw_hyperplane(ax, alpha, size=0.5):
    """Draw x+y+z=0 hyperplane with fade-in based on alpha."""
    if alpha < 0.3:
        return

    plane_alpha = min(0.25, (alpha - 0.3) * 0.5)

                                      
                      
    s = size
    verts = [
        [s, -s, 0], [s, 0, -s], [0, s, -s],
        [-s, s, 0], [-s, 0, s], [0, -s, s]
    ]

    ax.add_collection3d(Poly3DCollection(
        [verts], alpha=plane_alpha, facecolor='#90CAF9',
        edgecolor='#42A5F5', linewidth=1
    ))

                                                                               
                        
                                                                               
def draw_frame_3d(alpha, filename, title_text):
    """Generate a single static 3D frame."""
    fig = plt.figure(figsize=(5.5, 5.5), facecolor=COLORS['bg'])
    ax = fig.add_subplot(111, projection='3d')

                          
    if alpha > 0.5:
        setup_3d_axes(ax, lim=0.35)
        draw_hyperplane(ax, alpha, size=0.4)
    else:
        setup_3d_axes(ax, lim=1.15)

                                                   
                                                                         
    w1 = V1 - alpha * np.mean(V1)
    w2 = V2 - alpha * np.mean(V2)

                  
    draw_vector_3d(ax, w1, COLORS['solar'], 'A')
    draw_vector_3d(ax, w2, COLORS['glm'], 'B')

              
    draw_arc_3d(ax, w1, w2, COLORS['angle'])

                     
    current_cos = cos_sim(w1, w2)
    current_angle = angle_deg(w1, w2)

           
    ax.set_title(title_text, fontsize=13, fontweight='bold', color=COLORS['text'], pad=5)

                      
    info_text = f"cos = {current_cos:.2f}"
    ax.text2D(0.03, 0.93, info_text, transform=ax.transAxes, fontsize=11,
              va='top', ha='left', color=COLORS['text'],
              bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                       edgecolor='#CCCCCC', alpha=0.9))

              
    ax.text2D(0.95, 0.08, f"{current_angle:.0f}°", transform=ax.transAxes,
              fontsize=28, fontweight='bold', color=COLORS['angle'], alpha=0.3,
              ha='right', va='bottom')

                
    if alpha > 0.5:
                                   
        ax.view_init(elev=35.26, azim=45)
        ax.set_xlim(-0.15, 0.35)
        ax.set_ylim(-0.15, 0.35)
        ax.set_zlim(-0.15, 0.35)
    else:
        ax.view_init(elev=25, azim=135)

                        
    plt.savefig(filename, dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print(f"Saved: {filename}")

                                                                               
                     
                                                                               
def create_animation_3d(output_path, frames=120, fps=20):
    """Generate the 3D transition animation."""
    pause_start = 15              
    pause_end = 40                           
    transition = frames - pause_start - pause_end - 20         
    view_transition = 20         

    fig = plt.figure(figsize=(5.5, 5.5), facecolor=COLORS['bg'])
    ax = fig.add_subplot(111, projection='3d')

                                       
                                                   
    target_elev = 35.26
    target_azim = 45

    def update(frame):
        ax.cla()
        setup_3d_axes(ax, lim=1.15)

                                 
        if frame < pause_start:
            alpha, phase = 0.0, "initial"
        elif frame < pause_start + transition:
            t = (frame - pause_start) / transition
            alpha, phase = 0.5 - 0.5 * np.cos(np.pi * t), "transition"
        elif frame < pause_start + transition + view_transition:
            alpha, phase = 1.0, "view_transition"
        else:
            alpha, phase = 1.0, "final"

                                                                    
        draw_hyperplane(ax, alpha, size=0.4)

                                                       
                                                                             
        w1 = V1 - alpha * np.mean(V1)
        w2 = V2 - alpha * np.mean(V2)

                                                            
        arrow_ratio = 0.08 if np.linalg.norm(w1) > 0.3 else 0.15
        ax.quiver(0, 0, 0, w1[0], w1[1], w1[2],
                  color=COLORS['solar'], arrow_length_ratio=arrow_ratio, linewidth=3.5)
        ax.quiver(0, 0, 0, w2[0], w2[1], w2[2],
                  color=COLORS['glm'], arrow_length_ratio=arrow_ratio, linewidth=3.5)

                
        ax.text(w1[0] + 0.03, w1[1] + 0.03, w1[2] + 0.03,
                'A', fontsize=14, fontweight='bold', color=COLORS['solar'])
        ax.text(w2[0] + 0.03, w2[1] + 0.03, w2[2] + 0.03,
                'B', fontsize=14, fontweight='bold', color=COLORS['glm'])

                  
        draw_arc_3d(ax, w1, w2, COLORS['angle'])

                         
        current_cos = cos_sim(w1, w2)
        current_angle = angle_deg(w1, w2)

               
        titles = {
            "initial": "Raw Cosine Similarity",
            "final": "Pearson Correlation",
            "transition": "Subtracting Mean...",
            "view_transition": "Pearson Correlation"
        }
        ax.set_title(titles[phase], fontsize=13, fontweight='bold',
                    color=COLORS['text'], pad=5)

                   
        ax.text2D(0.03, 0.93, f"cos = {current_cos:.2f}", transform=ax.transAxes, fontsize=11,
                  va='top', ha='left', color=COLORS['text'],
                  bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                           edgecolor='#CCCCCC', alpha=0.9))

                  
        ax.text2D(0.95, 0.08, f"{current_angle:.0f}°", transform=ax.transAxes,
                  fontsize=28, fontweight='bold', color=COLORS['angle'],
                  alpha=0.2 + 0.15 * alpha, ha='right', va='bottom')

                    
        if phase == "initial" or phase == "transition":
                                 
            base_azim = 135 + frame * 0.3
            ax.view_init(elev=25, azim=base_azim)
        elif phase == "view_transition":
                           
            vt_frame = frame - (pause_start + transition)
            t = vt_frame / view_transition
            t_smooth = 0.5 - 0.5 * np.cos(np.pi * t)
            start_azim = 135 + (pause_start + transition) * 0.3
            current_elev = 25 + (target_elev - 25) * t_smooth
            current_azim = start_azim + (target_azim - start_azim) * t_smooth
            ax.view_init(elev=current_elev, azim=current_azim)
                          
            zoom = 0.3 + 0.7 * (1 - t_smooth)              
            lim = 1.15 * zoom
            ax.set_xlim(-0.3 * zoom, lim)
            ax.set_ylim(-0.3 * zoom, lim)
            ax.set_zlim(-0.3 * zoom, lim)
        else:
                                  
            ax.view_init(elev=target_elev, azim=target_azim)
                      
            zoom_lim = 0.35
            ax.set_xlim(-0.15, zoom_lim)
            ax.set_ylim(-0.15, zoom_lim)
            ax.set_zlim(-0.15, zoom_lim)

        return []

    print("Generating 3D animation...")
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

                                                                               
      
                                                                               
if __name__ == '__main__':
    draw_frame_3d(0, 'result/figure_00_cosine_before_3d.png', 'Raw Cosine Similarity')
    draw_frame_3d(1, 'result/figure_00_cosine_after_3d.png', 'Pearson Correlation')
    create_animation_3d('result/figure_00_cosine_bias_animation_3d.gif')
