import numpy as np

def computeHingePointNewTracker(xhead, yhead, xtail, ytail, xspine, yspine):
    if xhead-xtail == 0:
        m = (yhead-ytail)/(xhead+1-xtail);
    else:
        m = (yhead-ytail)/(xhead-xtail);
        
    n = (0.5)*((yhead+ytail)-np.multiply(m, (xhead+xtail)));
    sz = np.shape(xspine)[0]
    m2 = np.matlib.repmat(m, 1, sz);
    n2 = np.matlib.repmat(n, 1, sz);
    distances = np.abs(np.multiply(m, xspine) - yspine + n)/np.sqrt(np.square(m)+1);

    return np.argmax(distances)

def curvature_splines(x, y):
    dx  = np.gradient(x);
    ddx = np.gradient(dx);
    dy  = np.gradient(y);
    ddy = np.gradient(dy);
    
    num   = np.multiply(dx, ddy) - np.multiply(ddx, dy)
    denom = np.multiply(dx, dx) + np.multiply(dy, dy);
    denom = np.sqrt(denom);
    
    denom = denom * denom * denom;
    curvature = np.divide(num, denom);
    curvature[denom < 0] = np.nan;

    return curvature

def smooth(y, win_sz):
    box = np.ones(win_sz)/win_sz
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[-win_sz:] = y[-win_sz:].copy()
    y_smooth[:win_sz] = y[:win_sz].copy()
    
    return y_smooth

def get_bend_points(fcontour, head_index, tail_index):
    fcontour = np.roll(fcontour, -head_index, axis=0)
    tail_index = tail_index - head_index
    head_index = 0

    curve_1 = fcontour[head_index:tail_index, :].copy()
    curve_2 = fcontour[tail_index+1:, :].copy()
    del fcontour

    sub_points_1 = np.array(np.linspace(0, np.shape(curve_1)[0], 101, endpoint=False), dtype=np.int32)
    sub_points_2 = np.array(np.linspace(0, np.shape(curve_2)[0], 101, endpoint=False), dtype=np.int32)

    curve_1_sub = curve_1[sub_points_1, :]
    curve_2_sub = curve_2[sub_points_2, :]
    del sub_points_1, sub_points_2
    
    spine_win = 5
    curve_1_sub[:, 0] = smooth(curve_1_sub[:, 0], spine_win)
    curve_1_sub[:, 1] = smooth(curve_1_sub[:, 1], spine_win)
    curve_2_sub[:, 0] = smooth(curve_2_sub[:, 0], spine_win)
    curve_2_sub[:, 1] = smooth(curve_2_sub[:, 1], spine_win)    

    fspline = np.zeros(np.shape(curve_1_sub))
    fspline = (curve_1_sub + np.flip(curve_2_sub, axis=0))/2
    
    spine_win = 15
    fspline[:, 0] = smooth(fspline[:, 0], spine_win)
    fspline[:, 1] = smooth(fspline[:, 1], spine_win)
    
    fspline_crop = np.zeros(np.shape(fspline[7:-7, :]))
    fspline_crop[:, 0] = fspline[7:-7, 0].copy()
    fspline_crop[:, 1] = fspline[7:-7, 1].copy()
    del fspline

    xhead, yhead = fspline_crop[0, :].copy()
    xtail, ytail = fspline_crop[-1, :].copy()

    spline_inflec_index = computeHingePointNewTracker(xhead, yhead, xtail, ytail, fspline_crop[:, 0], fspline_crop[:, 1])
    spline_inflec_index = spline_inflec_index + spine_win
    
    gap = 3
    c1_pts = curve_1_sub[np.arange( spline_inflec_index-(gap*4),  spline_inflec_index+(gap*4), gap), :]
    c2_pts = curve_2_sub[np.arange(-spline_inflec_index-(gap*4), -spline_inflec_index+(gap*4), gap), :]

    curv_c1 = 1000000*np.nanmean(curvature_splines(c1_pts[:, 0], c1_pts[:, 1]))
    curv_c2 = 1000000*np.nanmean(curvature_splines(c2_pts[:, 0], c2_pts[:, 1]))

    c1_pts = np.hstack((c1_pts.copy(), curv_c1+np.zeros((np.shape(c1_pts)[0], 1))))
    c2_pts = np.hstack((c2_pts.copy(), curv_c2+np.zeros((np.shape(c2_pts)[0], 1))))

    final_mat = np.zeros((16,3), dtype=np.int64)
    if (curv_c1 <= 0) & (curv_c2 > 0):
        final_mat[:] = np.vstack((c1_pts, c2_pts))
    elif (curv_c2 <= 0) & (curv_c1 > 0):
        final_mat[:] = np.vstack((c2_pts, c1_pts))
    elif curv_c1 > curv_c2:
        final_mat[:] = np.vstack((c2_pts, c1_pts))
    elif curv_c2 > curv_c1:
        final_mat[:] = np.vstack((c1_pts, c2_pts))

    assert (np.shape(final_mat)[0] == 16) & (np.shape(final_mat)[1] == 3), 'wrong shape of the output matrix'
    
    return final_mat

