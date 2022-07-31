import numpy as np

def d2r(angle):
    return angle/180*np.pi

def rpy2rv(roll, pitch, yaw):
    alpha = d2r(yaw)
    beta = d2r(pitch)
    gamma = d2r(roll)

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)

    r11 = ca * cb
    r12 = ca * sb * sg - sa * cg
    r13 = ca * sb * cg + sa * sg
    r21 = sa * cb
    r22 = sa * sb * sg + ca * cg
    r23 = sa * sb * cg - ca * sg
    r31 = -sb
    r32 = cb * sg
    r33 = cb * cg

    theta = np.arccos((r11 + r22 + r33 - 1) / 2)
    sth = np.sin(theta)
    kx = (r32 - r23) / (2 * sth)
    ky = (r13 - r31) / (2 * sth)
    kz = (r21 - r12) / (2 * sth)

    return [(theta * kx), (theta * ky), (theta * kz)]


if __name__ == '__main__':

    # rpy = [3.123, -0.045, -2.063]  # 178.94, -2.6 133.91
    # r = 3.123
    # p = -0.045
    # y = -2.337
    r = 178.94
    p = -2.6
    y = 133.91

    print(rpy2rv(r,p,y))
    rx = -np.cos(y)*np.sin(p)*np.sin(r)-np.sin(y)*np.cos(r)
    print(rx)
    rx = np.cos(y)*np.cos(p)
    print(rx)


    print (np.sin(y))
    print(-(np.sin(r) * np.cos(y)))
    print(np.cos(r) * np.cos(p))