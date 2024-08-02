import numpy as np

def rotate_span2spc(span_vec):
    """
    This function rotates velocity data by -20 degrees about the x-axis in the span-i frame.

    NOTE: The convention for a -20 degree rotation is not clear to me at this point. 
          Traditionally, the rotation applied is counter-clockwise about an axis. 

    Parameters:
    -----------
    span_vec : array containing the velocity vector in SPAN-i instrument frame

    Returns:
    ROT_X : array with the SPAN-i data rotated into the s/c frame. 
        NOTE: Actual S/C frame: x --> z, y --> -x, z --> -y 
    """
    Rx = [[1, 0, 0],
          [0, np.cos(np.radians(-20)), -np.sin(np.radians(-20))],
          [0, np.sin(np.radians(-20)), np.cos(np.radians(-20))]]
    
    ROT_X = np.einsum('ij, tj->ti', Rx, span_vec)
    return(ROT_X)

def calc_flow_angles(velocity):
    """
    Standard cartesian to spherical calculation. Convert to elivaiton and azimuth.
    """
    # Get the x, y, z components
    x = velocity[:,0]
    y = velocity[:,1]
    z = velocity[:,2]

    # Convert to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(z/r)
    phi   = np.sign(y)*np.arccos(x/np.sqrt(x**2 + y**2))

    return(r, theta, phi)

def flow_angles_span_inst(span_vec):
    """
    Get flow angles in SPC frame from SPAN-i instrument frame.
    """
    span_rot = rotate_span2spc(span_vec)

    r, theta, phi = calc_flow_angles(span_rot)

    # Convert to SPC flow angles about the -z direction
    el = 90 - np.degrees(theta)
    az = np.degrees(phi) - 180

    return(el, az)