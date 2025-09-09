#utils.py
import numpy as np
import plotly.graph_objects as go

RADIATION_PATTERN_DEFAULTS = {
    "theta_3dB": 125,  # deg
    "phi_3dB": 125,    # deg
    "SLA_V": 22.5,     # dB
    "A_max": 22.5,     # dB
    "G_E_max": 5.3,    # dBi
    "boresight_to": (90, 0) # (theta0_deg, phi0_deg), (90,0) - along x-axis, (0,0) - along z-axis (up)
}

def reference_antenna_pattern(theta_in,
                              phi_in,
                              theta_3dB=RADIATION_PATTERN_DEFAULTS["theta_3dB"],
                              phi_3dB=RADIATION_PATTERN_DEFAULTS["phi_3dB"],
                              SLA_V=RADIATION_PATTERN_DEFAULTS["SLA_V"],
                              A_max=RADIATION_PATTERN_DEFAULTS["A_max"],
                              G_E_max=RADIATION_PATTERN_DEFAULTS["G_E_max"],
                              units="deg",
                              boresight_to=RADIATION_PATTERN_DEFAULTS["boresight_to"]):
    """
    3GPP TR 38.901 element pattern with optional boresight rotation.
    theta_in, phi_in: world-frame zenith/azimuth (broadcastable)
    units: "rad" or "deg" for the input angles
    boresight_to: (theta0_deg, phi0_deg) - boresight direction of the reference radiation pattern
    returns: A_ref_vert_dB, A_horiz_ref_dB, A_ref_dB, F_theta_ref_lin, F_phi_ref_lin
    """

    # angles -> degrees (for cuts) and radians (for rotation)
    if units == "rad":
        theta_deg = np.degrees(theta_in); phi_deg = np.degrees(phi_in)
        theta_rad = np.asarray(theta_in, float); phi_rad = np.asarray(phi_in, float)
    elif units == "deg":
        theta_deg = np.asarray(theta_in, float); phi_deg = np.asarray(phi_in, float)
        theta_rad = np.radians(theta_deg);       phi_rad = np.radians(phi_deg)
    else:
        raise ValueError("units must be 'rad' or 'deg'")

    theta_deg = np.clip(theta_deg, 0.0, 180.0)

    # rotate local element angles (θ″,φ″) if boresight is moved
    if boresight_to is not None:
        theta0_deg, phi0_deg = boresight_to
        a = np.radians(phi0_deg)               # yaw
        b = np.radians(theta0_deg - 90.0)      # pitch (FIXED SIGN)
        # g = 0 → no slant
        ct, st = np.cos(theta_rad), np.sin(theta_rad)
        cpa, spa = np.cos(phi_rad - a), np.sin(phi_rad - a)
        cb, sb = np.cos(b), np.sin(b)

        # γ=0 ⇒ (7.1-7) & (7.1-8) simplify to:
        cos_thp = cb*ct + sb*cpa*st
        cos_thp = np.clip(cos_thp, -1.0, 1.0)
        theta_p_deg = np.degrees(np.arccos(cos_thp))
        A = cb*st*cpa - sb*ct
        B = spa*st
        phi_p_deg = (np.degrees(np.arctan2(B, A)) + 180.0) % 360.0 - 180.0
    else:
        theta_p_deg = theta_deg
        phi_p_deg   = phi_deg ((phi_deg + 180.0) % 360.0) - 180.0

    # 3GPP vertical / horizontal cuts in local angles (θ″,φ″)
    A_ref_vert_dB  = -np.minimum(12.0 * ((theta_p_deg - 90.0) / theta_3dB) ** 2, SLA_V)
    A_horiz_ref_dB = -np.minimum(12.0 * (phi_p_deg / phi_3dB) ** 2, A_max)

    # combined attenuation (≤ 0 dB), capped by A_max plus total directional gain (dBi)
    A_ref_dB = np.maximum(A_ref_vert_dB + A_horiz_ref_dB, -A_max) + G_E_max

    # reference element field (θ-polarized), amplitude from attenuation
    F_theta_ref_lin = 10.0 ** (A_ref_dB / 20.0)
    F_phi_ref_lin   = np.zeros_like(F_theta_ref_lin)

    return A_ref_dB, F_theta_ref_lin, F_phi_ref_lin

def polarization_rotate(THETA_rad, PHI_rad,
                        alpha_deg, betta_deg, gamma_deg,
                        boresight_to=RADIATION_PATTERN_DEFAULTS["boresight_to"]):
    """
    Polarization rotation of the reference antenna pattern.
    THETA_rad, PHI_rad: meshgrid of angles in radians
    alpha_deg, betta_deg, gamma_deg: Euler angles of the antenna element rotation (deg)
    boresight_to: (theta0_deg, phi0_deg) - boresight
    returns: PSI_deg, F_theta, F_phi, A_db
    """
    a = np.radians(alpha_deg)
    b = np.radians(betta_deg)
    g = np.radians(gamma_deg)

    ct, st = np.cos(THETA_rad), np.sin(THETA_rad)
    cpa, spa = np.cos(PHI_rad - a), np.sin(PHI_rad - a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)

    # (7.1-7)
    cos_arg = cb*cg*ct + (sb*cg*cpa - sg*spa)*st
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    THETA_p = np.arccos(cos_arg)

    # (7.1-8)
    A = cb*st*cpa - sb*ct
    B = cb*sg*ct + (sb*sg*cpa + cg*spa)*st
    PHI_p = np.arctan2(B, A)

    # reference element pattern
    _, F_theta_p_ref, F_phi_p_ref = reference_antenna_pattern(THETA_p, PHI_p, units="rad", boresight_to=boresight_to)

    # ψ via unnormalized numerators (robust)
    num_cos = cb*cg*st - (sb*cg*cpa - sg*spa)*ct
    num_sin = sb*cg*spa + sg*cpa
    PSI = np.arctan2(num_sin, num_cos)
    PSI_deg = np.degrees(PSI)
    cosPsi, sinPsi = np.cos(PSI), np.sin(PSI)

    # (7.1-11)
    F_theta =  cosPsi*F_theta_p_ref - sinPsi*F_phi_p_ref
    F_phi   =  sinPsi*F_theta_p_ref + cosPsi*F_phi_p_ref

    A_lin = np.abs(F_theta)**2 + np.abs(F_phi)**2
    A_db  = 10*np.log10(A_lin + 1e-15)

    return PSI_deg, F_theta, F_phi, A_db

def UE_rotate (THETA_rad, PHI_rad,
               alpha_ue_deg, betta_ue_deg, gamma_ue_deg,
               alpha_deg, betta_deg, gamma_deg,
               boresight_to=RADIATION_PATTERN_DEFAULTS["boresight_to"]):
    """
    UE rotation of the reference antenna pattern.
    THETA_rad, PHI_rad: meshgrid of angles in radians
    alpha_ue_deg, betta_ue_deg, gamma_ue_deg: Euler angles of the UE rotation (deg)
    alpha_deg, betta_deg, gamma_deg: Euler angles of the antenna element rotation (deg)
    boresight_to: (theta0_deg, phi0_deg) - boresight
    returns: F_theta, F_phi, A_db
    """

    a = np.radians(alpha_ue_deg)
    b = np.radians(betta_ue_deg)
    g = np.radians(gamma_ue_deg)

    ct, st = np.cos(THETA_rad), np.sin(THETA_rad)
    cpa, spa = np.cos(PHI_rad - a), np.sin(PHI_rad - a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)

    # (7.1-7)
    cos_arg = cb*cg*ct + (sb*cg*cpa - sg*spa)*st
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    THETA_p = np.arccos(cos_arg)

    # (7.1-8)
    A = cb*st*cpa - sb*ct
    B = cb*sg*ct + (sb*sg*cpa + cg*spa)*st
    PHI_p = np.arctan2(B, A)

    #Firstly UE rotation
    _, F_theta_p_ref, F_phi_p_ref, _ = polarization_rotate(THETA_p, PHI_p, alpha_deg, betta_deg, gamma_deg, boresight_to)

    # ψ via unnormalized numerators (robust)
    num_cos = cb*cg*st - (sb*cg*cpa - sg*spa)*ct
    num_sin = sb*cg*spa + sg*cpa
    PSI = np.arctan2(num_sin, num_cos)
    #PSI_deg = np.degrees(PSI)
    cosPsi, sinPsi = np.cos(PSI), np.sin(PSI)

    # (7.1-11)
    F_theta =  cosPsi*F_theta_p_ref - sinPsi*F_phi_p_ref
    F_phi   =  sinPsi*F_theta_p_ref + cosPsi*F_phi_p_ref

    A_lin = np.abs(F_theta)**2 + np.abs(F_phi)**2
    A_db  = 10*np.log10(A_lin + 1e-15)

    return F_theta, F_phi, A_db

def plot_3d_pattern(THETA_rad,
                    PHI_rad,
                    Input, units,
                    title = '3D Pattern',
                    index="''"):
    """
    3D plot of the antenna radiation pattern using Plotly.
    THETA_rad, PHI_rad: meshgrid of angles in radians
    Input: A_db (if units="dB") or F_lin (if units="lin")
    units: "dB" or "lin"
    title: plot title
    index: string to append to axis labels, e.g. "''" or "'"
    returns: shows the 3D plot
    """

    if units == "dB":
        A_db = Input
        A_lin = 10 ** (A_db / 20)
        r = A_lin / np.max(A_lin)
        r = 0.1 + 0.9 * r  # avoid radius=0 for visualization
        C_db = A_db
    elif units == "lin":
        eps = 1e-15
        F_lin = Input
        P_lin = np.abs(F_lin)**2
        C_db = 10*np.log10(np.clip(P_lin, eps, None))   # ≤ 0 dB if fields are normalized
        r = np.sqrt(P_lin / (np.max(P_lin) + 1e-15))  # amplitude, normalized
    else:
        raise ValueError("units must be 'dB' or 'lin'")

    # Spherical to Cartesian coordinates
    x = r * np.sin(THETA_rad) * np.cos(PHI_rad)
    y = r * np.sin(THETA_rad) * np.sin(PHI_rad)
    z = r * np.cos(THETA_rad)

    # Surface plot
    surface = go.Surface(
        x=x, y=y, z=z,
        surfacecolor = C_db,
        colorscale='Turbo',
        cmax=5.5,
        cmin=-20.0,
        opacity=1,
        colorbar=dict(
            title=dict(
                text='Power [dB]',
                font=dict(size=18)
            ),
            tickfont=dict(size=16)
        ),
        showscale=True
    )

    # Axis length
    L = 1.2

    # Axis lines
    axes_lines = [
        go.Scatter3d(x=[0, L], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=4), showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, L], z=[0, 0], mode='lines', line=dict(color='black', width=4), showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, L], mode='lines', line=dict(color='black', width=4), showlegend=False)
    ]

    # Build figure
    fig = go.Figure(data=[surface] + axes_lines)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.0)),
            annotations=[
                dict(showarrow=False, x=L + 0.05, y=0, z=0, text="X"+index, xanchor="left", font=dict(size=18, color="black")),
                dict(showarrow=False, x=0, y=L + 0.05, z=0, text="Y"+index, xanchor="left", font=dict(size=18, color="black")),
                dict(showarrow=False, x=0, y=0, z=L + 0.05, text="Z"+index, xanchor="left", font=dict(size=18, color="black")),
            ],
        ),
        width=500,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def _R_zyx(alpha_deg, beta_deg, gamma_deg):
    a, b, g = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    Rz = np.array([[ca, -sa, 0],[sa, ca, 0],[0, 0, 1]])
    Ry = np.array([[cb, 0, sb],[0, 1, 0],[-sb, 0, cb]])
    Rx = np.array([[1, 0, 0],[0, cg, -sg],[0, sg, cg]])
    return Rz @ Ry @ Rx

def plot_patterns_on_phone(
        Input_list,
        THETA_rad, PHI_rad,
        units="lin",
        rect_size=(15.0, 7.0),
        angles_deg=(0.0, 0.0, 0.0),   # rotate plate & positions only
        positions=None,               # list of (x,y) on the plate; default = 8 around edges
        lobe_scale=1.0,
        cmin=-20.0, cmax=5.5,
        colorscale='Turbo',
        title='Fθ patterns around a rotated 15×7 plate'):
    
    Wx, Hy = rect_size
    # default 8 positions
    if positions is None:
        positions = [(-Wx/2, -Hy/2), (0.0, -Hy/2),
                     ( Wx/2, -Hy/2), (Wx/2, 0.0 ),
                     ( Wx/2, Hy/2 ), (0, Hy/2),
                     (-Wx/2, Hy/2), ( -Wx/2, 0)]
    positions = positions[:len(Input_list)]

    R = _R_zyx(*angles_deg)

    # plate (rotate)
    rect_x = np.array([[-Wx/2, Wx/2], [-Wx/2, Wx/2]])
    rect_y = np.array([[-Hy/2, -Hy/2], [Hy/2, Hy/2]])
    rect_z = np.zeros_like(rect_x)
    pts = np.stack([rect_x, rect_y, rect_z], axis=-1)
    pts_rot = pts @ R.T
    RX, RY, RZ = pts_rot[...,0], pts_rot[...,1], pts_rot[...,2]
    plate = go.Surface(x=RX, y=RY, z=RZ,
                       surfacecolor=np.zeros_like(RX),
                       colorscale=[[0,'#cccccc'], [1,'#cccccc']],
                       showscale=False, opacity=0.35)

    # precompute angle trig
    st, ct = np.sin(THETA_rad), np.cos(THETA_rad)
    cph, sph = np.cos(PHI_rad), np.sin(PHI_rad)

    surfaces = [plate]
    eps = 1e-15
    show_scale = True

    for Input, (px, py) in zip(Input_list, positions):
        if units == "dB":
            # power & color
            A_db = Input
            A_lin = 10 ** (A_db / 20)
            r = A_lin / np.max(A_lin)
            r = 0.1 + 0.9 * r
            r= lobe_scale * r  # tweak visual size of the lobes 
            C_db = A_db
        elif units == "lin":
            # power & color
            P_lin = np.abs(Input)**2
            C_db  = 10*np.log10(np.clip(P_lin, eps, None))
            # amplitude radius (per-lobe normalization)
            r = lobe_scale * np.sqrt(P_lin / (np.max(P_lin) + eps))
        else:
            raise ValueError("units must be 'dB' or 'lin'")

        # lobe geometry in WORLD frame (no extra rotation)
        xL = r * st * cph
        yL = r * st * sph
        zL = r * ct

        # rotate only the placement point (element location on the plate)
        p_world = R @ np.array([px, py, 0.0])
        Xw, Yw, Zw = xL + p_world[0], yL + p_world[1], zL + p_world[2]

        surfaces.append(go.Surface(
            x=Xw, y=Yw, z=Zw,
            surfacecolor=C_db,
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            showscale=show_scale,
            colorbar=dict(title='Power [dB]', tickfont=dict(size=14)) if show_scale else None,
            opacity=1.0
        ))
        show_scale = False  # one colorbar

    # world axes
    L = max(Wx, Hy) * 0.8
    axes = [
        go.Scatter3d(x=[0, L], y=[0, 0], z=[0, 0], mode='lines',
                     line=dict(color='black', width=4), showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, L], z=[0, 0], mode='lines',
                     line=dict(color='black', width=4), showlegend=False),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, L], mode='lines',
                     line=dict(color='black', width=4), showlegend=False),
    ]

    fig = go.Figure(data=surfaces + axes)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, showbackground=False, zeroline=False),
            aspectmode='data',
            annotations=[
                dict(showarrow=False, x=L + 0.05, y=0, z=0, text="X", xanchor="left", font=dict(size=18, color="black")),
                dict(showarrow=False, x=0, y=L + 0.05, z=0, text="Y", xanchor="left", font=dict(size=18, color="black")),
                dict(showarrow=False, x=0, y=0, z=L + 0.05, text="Z", xanchor="left", font=dict(size=18, color="black")),
            ],
            camera=dict(eye=dict(x=1.3, y=1.2, z=1.0)),
        ),
        width=600, height=500, margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.update_layout(scene=dict(
        domain=dict(x=[0.0, 0.97], y=[0.0, 1.0]),   # widen to the right edge
        aspectmode="data"                            # or "cube" to fill more uniformly
    ))

    # 3) keep the colorbar skinny and tucked on the right
    fig.update_traces(
        selector=dict(type="surface"),
        colorbar=dict(x=0.985, xanchor="left", len=0.9, thickness=12, outlinewidth=0))
    
    fig.show()