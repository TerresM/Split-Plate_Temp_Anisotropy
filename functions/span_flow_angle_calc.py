import numpy as np

def rotate_span2spc_old(span_vec):
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

def calc_flow_angles_old(velocity):
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

def flow_angles_span_inst2(span_vec):
    """
    Get flow angles in SPC frame from SPAN-i instrument frame.
    """
    span_rot = rotate_span2spc_old(span_vec)

    r, theta, phi = calc_flow_angles_old(span_rot)

    # Convert to SPC flow angles about the -z direction
    el = 90 - np.degrees(theta)
    az = np.degrees(phi) - 180

    return(el, az)

def rotate_span2spc(span_vec):
    """
    Rotates velocity data from the SPAN-i instrument frame directly into 
    the Spacecraft (SPC) frame using the exact SPICE matrix (Eq A.2).
    
    SPC Frame Geometry:
      +z: Directed towards the Sun
      +x: Directed in the ram direction (spacecraft motion)
      +y: Completes the right-handed triad
    """
    # Eq A.2 Rotation Matrix (SPAN-i -> SPC Spacecraft Frame)
    R_span_to_sc = np.array([
        [ 0.        , -0.9396926 , -0.34202015],
        [ 0.        ,  0.34202015, -0.9396926 ],
        [ 1.        ,  0.        ,  0.        ]
    ])
    
    # Efficiently multiply the 3x3 matrix across the (N, 3) time series of vectors
    ROT_SC = np.einsum('ij, tj->ti', R_span_to_sc, span_vec)
    
    return ROT_SC

def calc_flow_angles(velocity):
    """
    Calculates the SPC flow angles (deflection from anti-sunward flow).
    """
    x = velocity[:, 0]
    y = velocity[:, 1]
    z = velocity[:, 2]

    # The flow is moving AWAY from the Sun (-z direction)
    v_rad = -z 

    # Calculate flow angles in degrees (hovering around 0)
    # phi: Deflection in the x-z plane (Ram direction)
    # theta: Deflection in the y-z plane (Cross-ram direction)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.degrees(np.arctan2(x, v_rad))
        theta = np.degrees(np.arctan2(-y, v_rad))

    r = np.sqrt(x**2 + y**2 + z**2)

    return r, theta, phi

def flow_angles_span_inst(span_vec):
    """
    Get flow angles in SPC frame from SPAN-i instrument frame.
    """
    # 1. Rotate SPAN-i data DIRECTLY into the true Spacecraft (SPC) frame
    span_sc_frame = rotate_span2spc(span_vec)

    # 2. Calculate standard spherical coordinates about the +z (Sun) axis
    # Because we are in the true frame, no +/- 90 or 180 hacks are needed!
    r, theta, phi = calc_flow_angles(span_sc_frame)

    # Convert to SPC flow angles about the -z direction
    el = theta
    az = phi


    return(el, az)


