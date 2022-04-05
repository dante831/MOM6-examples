import numpy as np
from itertools import combinations, permutations

def coarse_grain_h(h, n_scale):
    # h is input field, with dimentions of t, z, y and x
    # n_scale is the coarse-grain factor
    if len(h.shape) == 2:
        h = h.reshape(1, 1, h.shape[0], h.shape[1])
    elif len(h.shape) == 3:
        h = h.reshape(1, h.shape[0], h.shape[1], h.shape[2])
    [nt, nz, ny0, nx0] = h.shape
    
    nxc = int(nx0/n_scale)
    nyc = int(ny0/n_scale)
    
    if nx0 == nxc and ny0 == nyc:
        return h
    else:
        return h.reshape(nt, nz, ny0, nxc, int(nx0/nxc)).mean(axis = 4)\
                .reshape(nt, nz, nyc, int(ny0/nyc), nxc).mean(axis = 3).squeeze()

def coarse_grain_u(u, n_scale):
    if n_scale == 1:
        return u
    # n_scale must be multiples of 2
    Nt, Nz, Ny, Nx_u = u.shape
    Nx = Nx_u - 1
    Nx_c = int(Nx/n_scale)
    Ny_c = int(Ny/n_scale)
    u_c = np.zeros((Nt, Nz, Ny_c, Nx_c+1))
    
    for t in range(Nt):
        if t % 100 == 0: print('t = {0}'.format(t)) 
        for k in range(Nz):
            for j in range(Ny_c):
                for i in range(Nx_c+1):
                    temp_N = 0
                    if i > 0:
                        u_c[t, k, j, i] += 0.5 * u[t, k, j*n_scale:(j+1)*n_scale, i*n_scale-int(n_scale/2)].sum()
                        temp_N += n_scale / 2
                    if i < Nx_c:
                        u_c[t, k, j, i] += 0.5 * u[t, k, j*n_scale:(j+1)*n_scale, i*n_scale+int(n_scale/2)].sum()
                        temp_N += n_scale / 2

                    lower = max(0, i*n_scale-int(n_scale/2)+1)
                    upper = min(Nx, i*n_scale+int(n_scale/2))
                    u_c[t, k, j, i] += u[t, k, j*n_scale:(j+1)*n_scale, lower:upper].sum()
                    temp_N += n_scale * (upper - lower)
                    u_c[t, k, j, i] /= temp_N
    return u_c

def coarse_grain_v(v, n_scale):
    if n_scale == 1:
        return v
    # n_scale must be multiples of 2
    Nt, Nz, Ny_v, Nx = v.shape
    Ny = Ny_v - 1
    Nx_c = int(Nx/n_scale)
    Ny_c = int(Ny/n_scale)
    v_c = np.zeros((Nt, Nz, Ny_c+1, Nx_c))
    
    for t in range(Nt):
        if t % 100 == 0: print('t = {0}'.format(t))
        for k in range(Nz):
            for j in range(Ny_c+1):
                for i in range(Nx_c):
                    temp_N = 0
                    if j > 0:
                        v_c[t, k, j, i] += 0.5 * v[t, k, j*n_scale-int(n_scale/2), i*n_scale:(i+1)*n_scale].sum()
                        temp_N += n_scale / 2
                    if j < Ny_c:
                        v_c[t, k, j, i] += 0.5 * v[t, k, j*n_scale+int(n_scale/2), i*n_scale:(i+1)*n_scale].sum()
                        temp_N += n_scale / 2

                    lower = max(0, j*n_scale-int(n_scale/2)+1)
                    upper = min(Ny, j*n_scale+int(n_scale/2))
                    v_c[t, k, j, i] += v[t, k, lower:upper, i*n_scale:(i+1)*n_scale].sum()
                    temp_N += n_scale * (upper - lower)
                    v_c[t, k, j, i] /= temp_N
    return v_c


def get_2d_index_h_to_staggered(r):
    r_scan = np.ceil(r).astype('int')
    u_ind, v_ind = [], []
    for j in range(-r_scan, r_scan+1):
        for i in range(-r_scan, r_scan+1):
            if np.sqrt((i-0.5)**2+j**2) <= r:
                u_ind.append(np.array([j, i]))
            if np.sqrt((j-0.5)**2+i**2) <= r:
                v_ind.append(np.array([j, i]))
    return u_ind, v_ind

def poly_features_uv_staggered(u, v, u_ind, v_ind, CROSS_TERMS = True, 
                     LINEAR_TERMS = False, max_inter_dis = float('inf')):

    if len(u_ind) != len(v_ind):
        print('Fatal error: u and v indices has different lengths!')
        return
    n_loc = len(u_ind)

    # Get boundary points 
    # (note that u has one extra point in x direction)
    # The same number of boundary point applies to v as well
    bnd = max([max([pair[0], pair[1]-1]) for pair in u_ind])

    if len(u.shape) == 2:
        for var in [u, v]:
            var = var.reshape(1, var.shape[0], var.shape[1])
    [nt, ny, nx_u] = u.shape; nx = nx_u - 1

    features = np.zeros([0, nt, ny-2*bnd, nx-2*bnd])
    var_dict = {'u': u, 'v': v}
    loc_indices = {'u': u_ind, 'v': v_ind}
    temp_X_feature = np.zeros([n_loc, nt, ny-2*bnd, nx-2*bnd])

    if CROSS_TERMS:
        comb_indices = list(combinations(np.arange(n_loc), 2))
        comb_indices = [pair for pair in comb_indices if abs(pair[0] - pair[1]) <= max_inter_dis]

        comb_indices_in_order = list(permutations(np.arange(n_loc), 2))
        comb_indices_in_order = [pair for pair in comb_indices_in_order 
                                     if abs(pair[0] - pair[1]) <= max_inter_dis]

        temp_X_feature_comb = np.zeros([len(comb_indices), nt, ny-2*bnd, nx-2*bnd])
        temp_X_feature_comb_in_order = np.zeros([len(comb_indices_in_order), nt, ny-2*bnd, nx-2*bnd])

    var_list = []
    ## Linear terms
    if LINEAR_TERMS:
        for key in ['u', 'v']:
            var = var_dict[key]
            for i in range(n_loc):
                temp_X_feature[i, :, :, :] = var[:, bnd+loc_indices[key][i][0]:ny-bnd+loc_indices[key][i][0], 
                                                    bnd+loc_indices[key][i][1]:nx-bnd+loc_indices[key][i][1]]
                var_list.append({key: [list(loc_indices[key][i])]})
            features = np.append(features, temp_X_feature, axis = 0)

    ## Quadratic terms
    for key in ['u', 'v']:
        var = var_dict[key]
        for i in range(n_loc):
            temp_X_feature[i, :, :, :] = var[:, bnd+loc_indices[key][i][0]:ny-bnd+loc_indices[key][i][0], 
                                                bnd+loc_indices[key][i][1]:nx-bnd+loc_indices[key][i][1]]**2
            var_list.append({key: [list(loc_indices[key][i]), list(loc_indices[key][i])]})
        features = np.append(features, temp_X_feature, axis = 0)

        # Cross terms
        if CROSS_TERMS:
            for i in range(len(comb_indices)):
                bias_0 = loc_indices[key][comb_indices[i][0]]
                bias_1 = loc_indices[key][comb_indices[i][1]]
                temp_X_feature_comb[i, :, :, :] = \
                            var[:, bnd+bias_0[0]:ny-bnd+bias_0[0], bnd+bias_0[1]:nx-bnd+bias_0[1]] * \
                            var[:, bnd+bias_1[0]:ny-bnd+bias_1[0], bnd+bias_1[1]:nx-bnd+bias_1[1]]
                var_list.append({key: [list(bias_0), list(bias_1)]})
            features = np.append(features, temp_X_feature_comb, axis = 0)

    # Interaction terms
    for key1, key2 in [['u', 'v']]:
        var1 = var_dict[key1]; var2 = var_dict[key2]
        for i in range(n_loc):
            temp_X_feature[i, :, :, :] = var1[:, bnd+loc_indices[key1][i][0]:ny-bnd+loc_indices[key1][i][0], 
                                                 bnd+loc_indices[key1][i][1]:nx-bnd+loc_indices[key1][i][1]] * \
                                         var2[:, bnd+loc_indices[key2][i][0]:ny-bnd+loc_indices[key2][i][0], 
                                                 bnd+loc_indices[key2][i][1]:nx-bnd+loc_indices[key2][i][1]]
            var_list.append({key1: [list(loc_indices[key][i])], key2: [list(loc_indices[key][i])]})
        features = np.append(features, temp_X_feature, axis = 0)

        # Cross terms 
        if CROSS_TERMS:
            for i in range(len(comb_indices_in_order)):
                bias_0 = loc_indices[key1][comb_indices_in_order[i][0]]
                bias_1 = loc_indices[key2][comb_indices_in_order[i][1]]
                temp_X_feature_comb_in_order[i, :, :, :] = \
                            var1[:, bnd+bias_0[0]:ny-bnd+bias_0[0], bnd+bias_0[1]:nx-bnd+bias_0[1]] * \
                            var2[:, bnd+bias_1[0]:ny-bnd+bias_1[0], bnd+bias_1[1]:nx-bnd+bias_1[1]]
                var_list.append({key1: [list(bias_0)], key2: [list(bias_1)]})
            features = np.append(features, temp_X_feature_comb_in_order, axis = 0)
    return features, var_list, bnd


def tendencies_from_subgrid(uu_sub, uv_sub, vv_sub, dx = 1, dy = 1):
    if len(uu_sub.shape) == 2:
        uu_sub = uu_sub[np.newaxis, np.newaxis, ...]
        uv_sub = uv_sub[np.newaxis, np.newaxis, ...]
        vv_sub = vv_sub[np.newaxis, np.newaxis, ...]
    elif len(uu_sub.shape) == 3:
        uu_sub = uu_sub[np.newaxis, ...]
        uv_sub = uv_sub[np.newaxis, ...]
        vv_sub = vv_sub[np.newaxis, ...]
    else:
        if len(uu_sub.shape) != 4:
            print('Error: input array must be 2, 3, or 4-dimensional')
            return None
    Nt, Nz, Ny, Nx = uu_sub.shape
    ## Tendencies
    # du
    du = np.zeros((Nt, Nz, Ny, Nx+1))
    du[:, :, :, 1:-1] = (uu_sub[:, :, :, 1:] - uu_sub[:, :, :, 0:-1])/dx
    du[:, :, 1:-1, 1:-1] += (uv_sub[:, :, 2:, 1:  ] - uv_sub[:, :, 0:-2, 1:  ] + 
                             uv_sub[:, :, 2:, 0:-1] - uv_sub[:, :, 0:-2, 0:-1])/(4*dy)
    du[:, :, 0, 1:-1] += (uv_sub[:, :, 1, 1:  ] - uv_sub[:, :, 0, 1:  ] + 
                          uv_sub[:, :, 1, 0:-1] - uv_sub[:, :, 0, 0:-1])/(2*dy)
    du[:, :, -1, 1:-1] += (uv_sub[:, :, -1, 1:  ] - uv_sub[:, :, -2, 1:  ] + 
                           uv_sub[:, :, -1, 0:-1] - uv_sub[:, :, -2, 0:-1])/(2*dy)
    # dv
    dv = np.zeros((Nt, Nz, Ny+1, Nx))
    dv[:, :, 1:-1, :] = (vv_sub[:, :, 1:, :] - vv_sub[:, :, 0:-1, :])/dy
    dv[:, :, 1:-1, 1:-1] += (uv_sub[:, :, 1:,   2:] - uv_sub[:, :, 1:  , 0:-2] + 
                             uv_sub[:, :, 0:-1, 2:] - uv_sub[:, :, 0:-1, 0:-2])/(4*dx)
    dv[:, :, 1:-1, 0] += (uv_sub[:, :, 1:,   1] - uv_sub[:, :, 1:  , 0] + 
                          uv_sub[:, :, 0:-1, 1] - uv_sub[:, :, 0:-1, 0])/(2*dx)
    dv[:, :, 1:-1, -1] += (uv_sub[:, :, 1:,   -1] - uv_sub[:, :, 1:  , -2] + 
                           uv_sub[:, :, 0:-1, -1] - uv_sub[:, :, 0:-1, -2])/(2*dx)
    return du.squeeze(), dv.squeeze()

def zb2020(u_c, v_c, dx = 1, dy = 1, kappa = -4.87e8, AZ2017 = False):

    ## ZB2020 parameterization for comparision
    ## Try calculating everything on h points in C-grid
    Nt, Nz, Ny_c, Nx_p1 = u_c.shape
    Nx_c = Nx_p1 - 1
    
    # Vorticity
    vort_cq = np.zeros((Nt, Nz, Ny_c+1, Nx_c+1))
    vort_cq[:, :, 1:-1, 1:-1] = (v_c[:, :, 1:-1, 1:] - v_c[:, :, 1:-1, 0:-1]) / dx \
                              - (u_c[:, :, 1:, 1:-1] - u_c[:, :, 0:-1, 1:-1]) / dy
        # Boundaries has zero vorticity
    vort_ch = 0.25 * (vort_cq[:, :, 0:-1, 0:-1] + vort_cq[:, :, 1:, 0:-1] + \
                      vort_cq[:, :, 0:-1, 1:  ] + vort_cq[:, :, 1:, 1:  ])

    # Divergence
    div_ch = (u_c[:, :, :, 1:] - u_c[:, :, :, 0:-1]) / dx + (v_c[:, :, 1:, :] - v_c[:, :, 0:-1, :]) / dy

    # Shear
    shear_cq = np.zeros((Nt, Nz, Ny_c+1, Nx_c+1))
    shear_cq[:, :, 1:-1, 1:-1] = (v_c[:, :, 1:-1, 1:] - v_c[:, :, 1:-1, 0:-1]) / dx \
                               + (u_c[:, :, 1:, 1:-1] - u_c[:, :, 0:-1, 1:-1]) / dy
    shear_ch = 0.25 * (shear_cq[:, :, 0:-1, 0:-1] + shear_cq[:, :, 1:, 0:-1] + \
                       shear_cq[:, :, 0:-1, 1:  ] + shear_cq[:, :, 1:, 1:  ])

    # Stretch
    stretch_ch = (u_c[:, :, :, 1:] - u_c[:, :, :, 0:-1]) / dx - (v_c[:, :, 1:, :] - v_c[:, :, 0:-1, :]) / dy

    # Subgrid-scale terms
    if AZ2017:
        uu_sub = - vort_ch * shear_ch
        vv_sub =   vort_ch * shear_ch
        uv_sub = vort_ch * stretch_ch
    else:
        uu_sub = 0.5 * ((vort_ch - shear_ch)**2 + stretch_ch**2)
        vv_sub = 0.5 * ((vort_ch + shear_ch)**2 + stretch_ch**2)
        uv_sub = vort_ch * stretch_ch

    du, dv = tendencies_from_subgrid(uu_sub, uv_sub, vv_sub, dx = dx, dy = dy)

    return kappa*du, kappa*dv, kappa*uu_sub, kappa*vv_sub, kappa*uv_sub
