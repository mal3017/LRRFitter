import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
import argparse
import os


from math import pi
from Bio.PDB import PDBParser
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize
from itertools import tee, islice, chain
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D

    # ----------------------------------------
    # User input
    # ----------------------------------------
def find_arguments():
    
    parser = argparse.ArgumentParser(description='Parameterising a protein circle.')

    parser.add_argument("Filename", help="Filename for PDB structure, .pdb does not need to be included")

    args = parser.parse_args()

    pdb_id = args.Filename

    filename = args.Filename
    file_extension = os.path.splitext(str(os.getcwd()) + "/" + filename)[1]
    if file_extension != ".pdb":
        filename = filename + ".pdb"
        
    starting_res = 1

    ending_res = 1000

    rep_unit = 1

    coil_len = 1
    
    return pdb_id, filename, starting_res, ending_res, rep_unit, coil_len

# ----------------------------------------
# 1: read coordinates - functions 
# ----------------------------------------

def read_and_clean_coordinates(filename, pdb_id, model_id = 0, chn_id = 'A'):
    """
    filename: (str) pdb structure file path
    pdb_id: (str) four letter pdb code
    model_id: (int) default to 0
    """
    
    # First read in the Ca coords
    
    p = PDBParser(PERMISSIVE=1)                    # PDB parser obj
    structure = p.get_structure(pdb_id, filename)  # read in PDB structure
    model = structure[model_id]                    # get model from structure
    chain = model[chn_id]
    
    # Clean PDB of all heteroatoms 
    
    residue_to_remove = []
    for residue in chain:
        if residue.id[0] != ' ':
            residue_to_remove.append((chain.id, residue.id))
    
    for residue in residue_to_remove:
        model[residue[0]].detach_child(residue[1])
        
    # Optional: remove loop regions
        
#    residue_ids_to_remove = [id no here]
    
#    for id in residue_ids_to_remove:
#        chain.detach_child((' ', id, ' '))
    
    return chain 

def get_ca_coord(chain, start, end):
    """
    chn_id: (str) default to 'A'
    index: (int) residue index starting from 0
    """
    Ca_num = len(chain)
    Cas = np.zeros((Ca_num, 3))
    atm_ca_list = []
    for i in range(Ca_num):
        res = chain.get_unpacked_list()[i] # returns list of all atoms
        atm_ca = res['CA'].get_vector()
        atm_ca_list = list(atm_ca)
        Cas[i] = atm_ca_list
        
    circle_region = Cas[start:(end + 1)]
        
    return circle_region


    
    # ----------------------------------------
    # User input
    # ----------------------------------------
    
pdb_id, filename, starting_res, ending_res, rep_unit, coil_len = find_arguments()
    
    # ----------------------------------------
    # 1: read coordinates - execution
    # ----------------------------------------
    
    # Read in the coordinates from the pdb file
PDB_coordinate_data = read_and_clean_coordinates(filename, pdb_id)
    
    # Get only the Ca coordiantes from the pdb file
P = get_ca_coord(PDB_coordinate_data, starting_res, ending_res)
print(P)
    

###########    









#PDB STUFF FIRST CIRCLE FITTING LATER








###############
    
def generate_circle_by_vectors(t, C, r, n, u):
    n = n/linalg.norm(n)
    u = u/linalg.norm(u)
    P_circle = r*cos(t)[:,newaxis]*u + r*sin(t)[:,newaxis]*cross(n,u) + C
    return P_circle
def generate_circle_by_angles(t, C, r, theta, phi):
    # Orthonormal vectors n, u, <n,u>=0
    n = array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    u = array([-sin(phi), cos(phi), 0])
    
    # P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
    P_circle = r*cos(t)[:,newaxis]*u + r*sin(t)[:,newaxis]*cross(n,u) + C
    return P_circle
#-------------------------------------------------------------------------------
# Plot
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# FIT CIRCLE 2D
# - Find center [xc, yc] and radius r of circle fitting to set of 2D points
# - Optionally specify weights for points
#
# - Implicit circle function:
#   (x-xc)^2 + (y-yc)^2 = r^2
#   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
#   c[0]*x + c[1]*y + c[2] = x^2+y^2
#
# - Solution by method of least squares:
#   A*c = b, c' = argmin(||A*c - b||^2)
#   A = [x y 1], b = [x^2+y^2]
#-------------------------------------------------------------------------------
def fit_circle_2d(x, y, w=[]):
    
    A = array([x, y, ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = diag(w)
        A = dot(W,A)
        b = dot(W,b)
    
    # Solve by method of least squares
    c = linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


#-------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(P, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/linalg.norm(n0)
    n1 = n1/linalg.norm(n1)
    k = cross(n0,n1)
    k = k/linalg.norm(k)
    theta = arccos(dot(n0,n1))
    
    # Compute rotated points
    P_rot = zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*cos(theta) + cross(k,P[i])*sin(theta) + k*dot(k,P[i])*(1-cos(theta))

    return P_rot


#-------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
#-------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return arctan2(linalg.norm(cross(u,v)), dot(u,v))
    else:
        return arctan2(dot(n,cross(u,v)), dot(u,v))

    
#-------------------------------------------------------------------------------
# - Make axes of 3D plot to have equal scales
# - This is a workaround to Matplotlib's set_aspect('equal') and axis('equal')
#   which were not working for 3D
#-------------------------------------------------------------------------------
def set_axes_equal_3d(ax):
    limits = array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = abs(limits[:,0] - limits[:,1])
    centers = mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])
    
#-------------------------------------------------------------------------------
# Init figures
#-------------------------------------------------------------------------------

alpha_pts = 0.5
figshape = (2,3)

#-------------------------------------------------------------------------------
# (1) Fitting plane by SVD for the mean-centered data
# Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
#-------------------------------------------------------------------------------
P_mean = P.mean(axis=0)
P_centered = P - P_mean
U,s,V = linalg.svd(P_centered)

# Normal vector of fitting plane is given by 3rd column in V
# Note linalg.svd returns V^T, so we need to select 3rd row from V^T
normal = V[2,:]
d = -dot(P_mean, normal)  # d = -<p,n>

#-------------------------------------------------------------------------------
# (2) Project points to coords X-Y in 2D plane
#-------------------------------------------------------------------------------
P_xy = rodrigues_rot(P_centered, normal, [0,0,1])

#-------------------------------------------------------------------------------
# (3) Fit circle in new 2D coords
#-------------------------------------------------------------------------------
xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])

#--- Generate circle points in 2D
t = linspace(0, 2*pi, 100)
xx = xc + r*cos(t)
yy = yc + r*sin(t)

#-------------------------------------------------------------------------------
# (4) Transform circle center back to 3D coords
#-------------------------------------------------------------------------------
C = rodrigues_rot(array([xc,yc,0]), [0,0,1], normal) + P_mean
C = C.flatten()

#--- Generate points for fitting circle
t = linspace(0, 2*pi, 100)
u = P[0] - C
P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

#--- Generate points for fitting arc
u = P[0] - C
v = P[-1] - C
theta = angle_between(u, v, normal)

t = linspace(0, theta, 100)
P_fitarc = generate_circle_by_vectors(t, C, r, normal, u)

print('Fitting circle: center = %s, r = %.4g angstroms' % (array_str(C, precision=4), r))

fig = figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.plot(*P.T, ls='', marker='o', alpha=0.5, label='Cluster points P')


#--- Plot fitting circle
ax.plot(*P_fitcircle.T, color='k', ls='--', lw=2, label='Fitting circle')
ax.plot(*P_fitarc.T, color='k', ls='-', lw=3, label='Fitting arc')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

ax.set_aspect('auto', 'datalim')
set_axes_equal_3d(ax)

plt.show()