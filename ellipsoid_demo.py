import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# ---------- Core function to create ellipsoid ----------
def make_ellipsoid(b_vec, w_para=1.0, w_perp=2.0):
    b_hat = np.array(b_vec)
    if np.allclose(b_hat, 0):
        b_hat = np.array([0, 0, 1])
    b_hat /= np.linalg.norm(b_hat)

    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    # Orthonormal basis around b_hat
    z_axis = b_hat
    temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    v_para = w_para * np.cos(theta)
    v_perp = w_perp * np.sin(theta)

    v = (v_para[..., None]*z_axis +
         v_perp[..., None]*(np.cos(phi)[..., None]*x_axis +
                            np.sin(phi)[..., None]*y_axis))
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # ---------- Compute centroids along z ----------
    dz = 0.02 * (vz.max() - vz.min())
    z_levels = np.linspace(vz.min(), vz.max(), 10)
    centroids = []
    for z0 in z_levels:
        mask = np.abs(vz - z0) < dz
        if np.any(mask):
            cx = vx[mask].mean()
            cy = vy[mask].mean()
            cz = vz[mask].mean()
            centroids.append((cx, cy, cz))
    centroids = np.array(centroids)

    # Create figure
    fig = go.Figure()

    # Ellipsoid surface
    fig.add_trace(go.Surface(
        x=vx, y=vy, z=vz,
        opacity=0.7, colorscale='Greys', showscale=False,
        name='Gyrotropic Ellipsoid'
    ))

    # Magnetic field direction
    fig.add_trace(go.Scatter3d(
        x=[0, 3*b_hat[0]], y=[0, 3*b_hat[1]], z=[0, 3*b_hat[2]],
        mode='lines+markers',
        line=dict(color='red', width=6),
        name='Magnetic Field (b̂)'
    ))

    # ✅ NEW: Projection of B onto the x–z plane (set y = -3)
    fig.add_trace(go.Scatter3d(
        x=[0, 3*b_hat[0]], 
        y=[-3, -3],
        z=[0, 3*b_hat[2]],
        mode='lines+markers',
        line=dict(color='rgba(255,0,0,0.5)', width=4, dash='dash'),
        marker=dict(size=3, color='rgba(255,0,0,0.5)'),
        name='B̂ projection (x–z plane)'
    ))

    # ✅ NEW: Projection of B onto the y–z plane (set x = -3)
    fig.add_trace(go.Scatter3d(
        x=[-3, -3],
        y=[0, 3*b_hat[1]],
        z=[0, 3*b_hat[2]],
        mode='lines+markers',
        line=dict(color='rgba(255,0,0,0.5)', width=4, dash='dash'),
        marker=dict(size=3, color='rgba(255,0,0,0.5)'),
        name='B̂ projection (y–z plane)'
    ))

    # Projection on x-z plane
    fig.add_trace(go.Surface(
        x=vx, y=np.zeros_like(vx) - 3, z=vz,
        opacity=0.3, colorscale='Blues', showscale=False,
        name='x-z Projection'
    ))

    # Projection on x-y plane
    fig.add_trace(go.Surface(
        x=vx, y=vy, z=np.zeros_like(vx) - 3,
        opacity=0.3, colorscale='Reds', showscale=False,
        name='x-y Projection'
    ))

    # ✅ NEW: Projection on y-z plane (x = -3)
    fig.add_trace(go.Surface(
        x=np.zeros_like(vx) - 3, y=vy, z=vz,
        opacity=0.3, colorscale='Greens', showscale=False,
        name='y-z Projection'
    ))

     # ---------- Centroid path ----------
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
        mode='markers+lines',
        marker=dict(size=5, color='black'),
        line=dict(color='black', width=2),
        name='Centroid Path'
    ))

    # Centroid projection onto x–z (y=-3)
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=np.zeros_like(centroids[:, 1]) - 3,
        z=centroids[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color='blue'),
        line=dict(color='blue', width=1),
        name='Centroid proj (x–z)'
    ))

    # Centroid projection onto y–z (x=3)
    fig.add_trace(go.Scatter3d(
        x=np.zeros_like(centroids[:, 0]) - 3,
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color='green'),
        line=dict(color='green', width=1),
        name='Centroid proj (y–z)'
    ))
    # ---------- Vertical reference line with arrow ----------
    z_arrow_top = 3.5  # height of arrow tip
    z_arrow_base = 3.0  # where arrow tip starts

    # Line (shaft)
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, z_arrow_base],
        mode='lines',
        line=dict(color='black', width=4, dash='dash'),
        name='z-axis'
    ))

    # Arrow tip (cone)
    fig.add_trace(go.Cone(
        x=[0],
        y=[0],
        z=[z_arrow_base],
        u=[0],
        v=[0],
        w=[1],  # points in +z
        colorscale=[[0, "black"], [1, "black"]],
        showscale=False,
        sizemode="absolute",
        sizeref=0.4,
        anchor="tail",
        name='z-axis arrow'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='v_x',
            yaxis_title='v_y',
            zaxis_title='v_z',
            aspectmode='data'
        ),
        title=f"Gyrotropic Ellipsoid (Bx={b_vec[0]:.2f}, By={b_vec[1]:.2f}, Bz={b_vec[2]:.2f})",
        width=800, height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-3.2, 3.2]),  # Sets x-axis range for the 3D scene
        yaxis=dict(range=[-3.2, 3.2]),  # Sets y-axis range for the 3D scene
        zaxis=dict(range=[-3.2, 3.2])   # Sets z-axis range for the 3D scene
    )
)

    return fig

# ---------- Dash app ----------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Gyrotropic Ellipsoid Visualization"),
    dcc.Graph(id='ellipsoid-plot', figure=make_ellipsoid([0.1, 0.2, 0.85])),

    html.Div([
        html.Label("Bx"),
        dcc.Slider(id='bx', min=-1.0, max=1.0, step=0.1, value=0.1,
                   marks={i: str(i) for i in np.linspace(-1, 1, 5)}),
        html.Label("By"),
        dcc.Slider(id='by', min=-1.0, max=1.0, step=0.1, value=0.2,
                   marks={i: str(i) for i in np.linspace(-1, 1, 5)}),
        html.Label("Bz"),
        dcc.Slider(id='bz', min=-1.0, max=1.0, step=0.1, value=0.85,
                   marks={i: str(round(i, 1)) for i in np.linspace(-1.0, 1.0, 5)})
    ], style={'width': '80%', 'margin': 'auto'})
])

# ---------- Callback to update plot ----------
@app.callback(
    Output('ellipsoid-plot', 'figure'),
    [Input('bx', 'value'),
     Input('by', 'value'),
     Input('bz', 'value')]
)
def update_ellipsoid(bx, by, bz):
    return make_ellipsoid([bx, by, bz])

# ---------- Run app ----------
if __name__ == '__main__':
    app.run(debug=True)
