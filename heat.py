from build_graph import create_adjacency_matrix
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def animate_low_high_freq(mesh, X_heat, U, Lambda, low_ratio=0.2, high_ratio=0.2, interval=200):
    """
    Animate low- and high-frequency components of heat propagation on a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh object.
    X_heat : torch.Tensor [num_nodes, T]
        Heat over time.
    U : torch.Tensor [num_nodes, num_nodes]
        Laplacian eigenvectors.
    Lambda : torch.Tensor [num_nodes]
        Laplacian eigenvalues.
    low_ratio : float
        Fraction of eigenvectors to use for low-frequency component.
    high_ratio : float
        Fraction of eigenvectors to use for high-frequency component.
    interval : int
        Delay between frames in milliseconds.
    """
    vertices = mesh.vertices
    num_nodes = U.shape[0]
    num_steps = X_heat.shape[1]

    # Precompute low/high frequency bases
    sorted_indices = torch.argsort(Lambda)
    n_low = int(num_nodes * low_ratio)
    n_high = int(num_nodes * high_ratio)
    U_low = U[:, sorted_indices[:n_low]]
    U_high = U[:, sorted_indices[-n_high:]]

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    scat_low = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=X_heat[:, 0].cpu().numpy(), cmap='hot',
                           s=20)
    scat_high = ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=X_heat[:, 0].cpu().numpy(), cmap='hot',
                            s=20)
    ax1.set_title("Low-frequency Heat")
    ax2.set_title("High-frequency Heat")

    def update(t):
        x_t = X_heat[:, t]

        x_low = U_low @ (U_low.T @ x_t)
        x_high = U_high @ (U_high.T @ x_t)

        scat_low.set_array(x_low.cpu().numpy())
        scat_high.set_array(x_high.cpu().numpy())
        ax1.set_title(f"Low-frequency Heat (t={t})")
        ax2.set_title(f"High-frequency Heat (t={t})")
        return scat_low, scat_high

    anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=False)
    plt.show()
    return anim


def visualize_low_high_freq(mesh, X_heat, U, Lambda, step=0, low_ratio=0.2, high_ratio=0.2):
    """
    Visualize low- and high-frequency components of the heat on the mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh object.
    X_heat : torch.Tensor [num_nodes, T]
        Heat over time.
    U : torch.Tensor [num_nodes, num_nodes]
        Laplacian eigenvectors.
    Lambda : torch.Tensor [num_nodes]
        Laplacian eigenvalues.
    step : int
        Timestep to visualize.
    low_ratio : float
        Fraction of eigenvectors to use for low-frequency component.
    high_ratio : float
        Fraction of eigenvectors to use for high-frequency component.
    """
    vertices = mesh.vertices
    num_nodes = U.shape[0]

    # Sort eigenvalues
    sorted_indices = torch.argsort(Lambda)

    # Low-frequency projection
    n_low = int(num_nodes * low_ratio)
    U_low = U[:, sorted_indices[:n_low]]
    x_low = U_low @ (U_low.T @ X_heat[:, step])

    # High-frequency projection
    n_high = int(num_nodes * high_ratio)
    U_high = U[:, sorted_indices[-n_high:]]
    x_high = U_high @ (U_high.T @ X_heat[:, step])

    # Plot
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                c=x_low.cpu().numpy(), cmap='hot', s=20)
    ax1.set_title("Low-frequency Heat")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                c=x_high.cpu().numpy(), cmap='hot', s=20)
    ax2.set_title("High-frequency Heat")

    plt.show()


def farthest_point_sampling(vertices, k):
    """
    Approximate evenly-spaced sampling of mesh vertices using FPS.

    vertices : (N, 3) tensor or numpy array
    k : number of samples to pick

    Returns:
        indices of k sampled vertices
    """
    verts = torch.tensor(vertices, dtype=torch.float32)
    N = verts.shape[0]

    # Pick a random starting point
    idx = torch.randint(0, N, (1,))
    sampled = [idx.item()]

    # Distances from selected points
    dist = torch.full((N,), float('inf'))

    for _ in range(1, k):
        # Update distance to nearest sampled point
        last = sampled[-1]
        dist = torch.minimum(
            dist,
            torch.norm(verts - verts[last], dim=1)
        )

        # Choose the vertex farthest from all chosen points
        next_idx = torch.argmax(dist).item()
        sampled.append(next_idx)

    return sampled


def simulate_heat_with_even_sensors(mesh, num_sensors, steps=20, alpha=0.4):
    """
    Simulate heat diffusion on mesh, but return only sensor values.
    Sensors are chosen using farthest-point sampling (uniformly on surface).
    """

    # Choose evenly spaced sensors
    sensor_indices = farthest_point_sampling(mesh.vertices, num_sensors)

    num_nodes = len(mesh.vertices)
    _, L = create_adjacency_matrix(mesh)

    # Initial condition – heat at first sensor
    x = torch.zeros(num_nodes, 1)
    # x[sensor_indices[0]] = 1.0
    # print(sensor_indices[0])
    x[897] = 1.0

    X_seq = [x.clone()]

    for _ in range(steps):
        x = x - alpha * (L @ x)
        X_seq.append(x.clone())

    X_seq_full = torch.cat(X_seq, dim=1)

    # Masked version (others = 0)
    X_seq_nan = X_seq_full.clone()
    mask = torch.ones(num_nodes, dtype=bool)
    mask[sensor_indices] = False
    X_seq_nan[mask] = 0.0

    return X_seq_nan, X_seq_full, sensor_indices


def simulate_heat_with_sensors(mesh, sensor_indices, steps=20, alpha=0.4):
    """
    Simulate heat diffusion on the mesh but only return values
    for selected sensor nodes; all other nodes are NaN.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The geometry.
    sensor_indices : list[int]
        Indices of vertices where sensors exist.
    steps : int
        Number of time steps to simulate.
    alpha : float
        Diffusion coefficient.

    Returns
    -------
    X_seq_nan : tensor [num_nodes, steps+1]
        Heat values; non-sensor nodes replaced with NaN.
    X_seq_full : tensor [num_nodes, steps+1]
        Full underlying (unmasked) heat field.
    """
    num_nodes = len(mesh.vertices)
    _, L = create_adjacency_matrix(mesh)

    # Initial heat distribution
    x = torch.zeros(num_nodes, 1)
    x[sensor_indices[0]] = 1.0  # initial heat at the first sensor

    X_seq = [x.clone()]

    # Heat diffusion process
    for _ in range(steps):
        x = x - alpha * (L @ x)
        X_seq.append(x.clone())

    # Full underlying field
    X_seq_full = torch.cat(X_seq, dim=1)

    # Create masked copy
    X_seq_nan = X_seq_full.clone()

    # Mask all non-sensor nodes
    all_nodes = torch.arange(num_nodes)
    mask = ~torch.isin(all_nodes, torch.tensor(sensor_indices))
    # X_seq_nan[mask, :] = float("nan")
    X_seq_nan[mask, :] = 0
    return X_seq_nan, X_seq_full


def simulate_heat(mesh, steps=20, alpha=0.4):
    num_nodes = len(mesh.vertices)
    _, L = create_adjacency_matrix(mesh)
    x = torch.zeros(num_nodes, 1)
    x[0] = 1.0  # heat source at first vertex
    X_seq = [x.clone()]
    for _ in range(steps):
        x = x - alpha * L @ x
        X_seq.append(x.clone())
    return torch.cat(X_seq, dim=1)


import torch


def simulate_heat_2(mesh, steps=20, alpha=0.4):
    num_nodes = len(mesh.vertices)
    _, L = create_adjacency_matrix(mesh)

    x = torch.zeros(num_nodes, 1)

    # choose two random distinct vertices for heat sources
    heat_indices = torch.randperm(num_nodes)[:2]
    x[heat_indices, 0] = 1.0

    X_seq = [x.clone()]

    for _ in range(steps):
        x = x - alpha * L @ x
        X_seq.append(x.clone())

    return torch.cat(X_seq, dim=1)


def visualize_heat(mesh, x_pred_seq):
    num_nodes = len(mesh.vertices)
    vertices = mesh.vertices
    # Plot heat propagation at steps 0, 5, 10, 20
    fig = plt.figure(figsize=(12, 3))
    for i, t in enumerate([0, 5, 10, 20]):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c=x_pred_seq[:, t].numpy(), cmap='hot', s=20)
        ax.set_title(f"Step {t}")
    plt.show()


def visualize_heat_single_step(mesh, x_pred_seq, timestep=5):
    """
    Visualize the heat on the mesh at a single timestep with edges.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh object with vertices and faces.
    x_pred_seq : torch.Tensor [num_nodes, steps]
        Heat values over time.
    timestep : int
        Timestep to visualize.
    """
    vertices = mesh.vertices
    adj, _ = create_adjacency_matrix(mesh)

    # Get edges from adjacency
    edges = torch.nonzero(adj)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    c=x_pred_seq[:, timestep].numpy(), cmap='hot', s=40)

    # Plot edges
    for u, v in edges:
        x = [vertices[u, 0], vertices[v, 0]]
        y = [vertices[u, 1], vertices[v, 1]]
        z = [vertices[u, 2], vertices[v, 2]]
        ax.plot(x, y, z, color='gray', alpha=0.3)

    # Add colorbar
    fig.colorbar(sc, ax=ax, label='Heat')

    ax.set_title(f"Heat on Mesh at Timestep {timestep}")
    ax.axis('off')
    plt.show()


def visualize_heat_single_step_zoom(mesh, x_pred_seq, timestep=5, zoom_fraction=0.2):
    """
    Visualize the heat on the mesh at a single timestep with edges,
    zooming into a subset that includes the hottest point.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh object with vertices and faces.
    x_pred_seq : torch.Tensor [num_nodes, steps]
        Heat values over time.
    timestep : int
        Timestep to visualize.
    zoom_fraction : float
        Fraction of the bounding box to zoom into (0 < zoom_fraction <= 1).
    """
    vertices = mesh.vertices
    adj, _ = create_adjacency_matrix(mesh)
    edges = torch.nonzero(adj)

    # Find the hottest node at this timestep
    hottest_idx = torch.argmax(x_pred_seq[:, timestep])

    # Compute bounding box around the hottest node
    center = vertices[hottest_idx]
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    range_xyz = max_corner - min_corner
    zoom_radius = zoom_fraction * range_xyz  # zoom around hottest point

    # Select vertices within zoom box centered on hottest point
    mask = (
            (vertices[:, 0] >= center[0] - zoom_radius[0] / 2) & (vertices[:, 0] <= center[0] + zoom_radius[0] / 2) &
            (vertices[:, 1] >= center[1] - zoom_radius[1] / 2) & (vertices[:, 1] <= center[1] + zoom_radius[1] / 2) &
            (vertices[:, 2] >= center[2] - zoom_radius[2] / 2) & (vertices[:, 2] <= center[2] + zoom_radius[2] / 2)
    )

    # Ensure hottest node is included
    mask[hottest_idx] = True

    vertices_sub = vertices[mask]
    heat_sub = x_pred_seq[mask, timestep]

    # Filter edges where both endpoints are in the subset
    edges_sub = torch.tensor([[u, v] for u, v in edges if mask[u] and mask[v]])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    sc = ax.scatter(vertices_sub[:, 0], vertices_sub[:, 1], vertices_sub[:, 2],
                    c=heat_sub.numpy(), cmap='hot', s=40)

    # Plot edges
    for u, v in edges_sub:
        x = [vertices[u, 0], vertices[v, 0]]
        y = [vertices[u, 1], vertices[v, 1]]
        z = [vertices[u, 2], vertices[v, 2]]
        ax.plot(x, y, z, color='gray', alpha=0.3)

    # Colorbar and title
    # fig.colorbar(sc, ax=ax, label='Heat')
    # ax.set_title(f"Heat on Mesh at Timestep {timestep}")
    ax.axis('off')
    plt.show()


def visualize_heat_with_sensors(mesh, x_pred_seq, sensor_indices, steps_to_plot=None):
    """
    Visualize heat propagation and sensor positions on a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh geometry.
    x_pred_seq : torch.Tensor [num_nodes, T]
        Heat values over time.
    sensor_indices : list or array
        Indices of sensor nodes.
    steps_to_plot : list[int] or None
        Timesteps to visualize. Defaults to [0, 5, 10, 20].
    """

    vertices = mesh.vertices
    num_nodes = len(vertices)
    steps_to_plot = steps_to_plot or [0, 5, 10, 20]

    fig = plt.figure(figsize=(12, 3))
    for i, t in enumerate(steps_to_plot):
        ax = fig.add_subplot(1, len(steps_to_plot), i + 1, projection='3d')
        # plot all nodes colored by heat
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c=x_pred_seq[:, t].cpu().numpy(), cmap='hot', s=20)
        # overlay sensors in blue
        ax.scatter(vertices[sensor_indices, 0],
                   vertices[sensor_indices, 1],
                   vertices[sensor_indices, 2],
                   c='blue', s=40, label='Sensors')
        ax.set_title(f"Step {t}")
        ax.legend()
    plt.show()


def plot_mesh_signal(mesh, signal, title=""):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    triangles = mesh.faces
    # color by signal at vertices
    p = ax.plot_trisurf(x, y, triangles, z, cmap='coolwarm', shade=True, linewidth=0.2, antialiased=True)
    p.set_array(signal)
    p.autoscale()
    fig.colorbar(p, ax=ax)
    ax.set_title(title)
    ax.view_init(elev=30, azim=45)
    ax.axis('off')
    plt.show()


def visualize_heat_at_sensors(mesh, x_pred_seq, sensor_indices, steps_to_plot=None):
    """
    Visualize heat propagation only at the sensor positions on a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh geometry.
    x_pred_seq : torch.Tensor [num_nodes, T]
        Heat values over time.
    sensor_indices : list or array
        Indices of sensor nodes.
    steps_to_plot : list[int] or None
        Timesteps to visualize. Defaults to [0, 5, 10, 20].
    """
    vertices = mesh.vertices
    steps_to_plot = steps_to_plot or [0, 5, 10, 20]

    fig = plt.figure(figsize=(12, 3))

    for i, t in enumerate(steps_to_plot):
        ax = fig.add_subplot(1, len(steps_to_plot), i + 1, projection='3d')

        # plot only sensor nodes colored by their heat
        ax.scatter(vertices[sensor_indices, 0],
                   vertices[sensor_indices, 1],
                   vertices[sensor_indices, 2],
                   c=x_pred_seq[sensor_indices, t].cpu().numpy(),
                   cmap='hot', s=40)

        ax.set_title(f"Step {t} (Sensors)")

    plt.show()
