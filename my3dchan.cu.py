import numpy as np 
from numba import cuda
from numba import vectorize, float64

TPB = (8,8,8)  # threads per block (tuned for V100)


N = 17  # must be odd for a point on the centerline
NITER = 25
NU = 1/180
dt = 0.000001
timesteps = 10000
CFL = 0.1


@cuda.jit
def solve_uvw( dt,dx,y,dz,u,v,w,rhs_u,rhs_v,rhs_w, ncv_x, ncv_y, ncv_z, nno_x, nno_y, nno_z ):

    k, j, i = cuda.grid(3)

    # x-momentum (final terms land in ucv-centers)
    if (i >= 1 and i < nno_x-1) and (j >= 1 and j < ncv_y-1) and (k >= 1 and k < ncv_z-1):

        # pre-compute useful velocities
        # compute neighbouring CV quantities
        ucv  = u[k,j,i]
        ucvE = u[k,j,i+1]
        ucvW = u[k,j,i-1]
        ucvN = u[k,j+1,i]
        ucvS = u[k,j-1,i]
        ucvF = u[k+1,j,i]
        ucvB = u[k-1,j,i]
        # compute CV corner quantities
        vfNW = v[k,j,i]
        vfSW = v[k,j-1,i]
        vfNE = v[k,j,i+1]
        vfSE = v[k,j-1,i+1]

        wfFW = w[k,j,i]
        wfBW = w[k-1,j,i]
        wfFE = w[k,j,i+1]
        wfBE = w[k-1,j,i+1]

        # convective terms:
        # d(uu)/dx +d(vu)/dy +d(wu)/dz
        conv = 0.0

        # d(uu)/dx
        # interp u in x so it is at E and W faces
        ufE = 0.5*( ucv +ucvE )
        ufW = 0.5*( ucvW +ucv )

        conv += ( ufE*ufE -ufW*ufW ) / dx

        # d(vu)/dy
        # interp v in x so it is at N and S faces
        vfN = 0.5*( vfNW +vfNE )
        vfS = 0.5*( vfSW +vfSE )
        # compute required ucv lengths
        dy = y[j+1] -y[j]
        dyN = y[j+2] -y[j+1]
        dyS = y[j] -y[j-1]
        # interp u in y so it is at N and S faces
        ufN = ( (dyN/2)*ucv +(dy/2)*ucvN ) / (dyN/2 +dy/2)
        ufS = ( (dy/2)*ucvS +(dyS/2)*ucv ) / (dy/2 +dyS/2)

        conv += ( vfN*ufN -vfS*ufS ) / dy

        # d(wu)/dz
        # interp w in x so it is at F and B faces
        wfF = 0.5*( wfFW +wfFE )
        wfB = 0.5*( wfBW +wfBE )
        # interp u in z so it is at F and B faces
        ufF = 0.5*( ucv +ucvF )
        ufB = 0.5*( ucvF +ucv )
        conv += ( wfF*ufF -wfB*ufB ) / dz

        # diffusive terms:
        # d2(u)/dx2 +d2(u)/dy2 +d2(u)/dz2
        diff = 0.0

        # d2(u)dx2 = d/dx(du/dx)
        # d2(u)/dx2
        # compute dudx (lands on E/W faces)
        dudxE = ( ucvE -ucv ) / dx
        dudxW = ( ucv -ucvW ) / dx
        # compute d2udx2
        diff += NU*( dudxE -dudxW ) / dx

        # d2(u)dy2
        # compute dudy (lands on N/S faces)
        PFN = (dyN/2)/(dy/2)
        # see Sundqvist and Veronis (1969) eq 1.3
        dudyN = ( ucvN-(PFN**2)*ucv -(1 -(PFN**2))*ufN ) /\
                (dyN/2*(1+PFN))
        PFS = (dy/2)/(dyS/2)
        dudyS = (ucv-(PFS**2)*ucvS -(1 -(PFS**2))*ufS) /\
                (dy/2*(1+PFS))

        diff += NU*( dudyN -dudyS ) / dy

        # d2(u)dz2
        # compute dudz (lands on F/B faces)
        dudzF = ( ucvF -ucv ) / dz
        dudzB = ( ucv -ucvB ) / dz
        # compute d2udz2
        diff += NU*( dudzF -dudzB ) / dz

        rhs_u[k,j,i] =  -conv +diff +1.0

    # y-momentum (final terms land in vcv-centers)
    if (i>=1 and i< ncv_x-1 ) and (j >= 1 and j < nno_y-1) and (k>=1 and k < ncv_z-1):

        # compute neighbouring CV quantities
        vcv  = v[k,j,i]
        vcvE = v[k,j,i+1]
        vcvW = v[k,j,i-1]
        vcvN = v[k,j+1,i]
        vcvS = v[k,j-1,i]
        vcvF = v[k+1,j,i]
        vcvB = v[k-1,j,i]
        # compute CV corner quantities
        ufNW = u[k,j+1,i-1]
        ufSW = u[k,j,i-1]
        ufNE = u[k,j+1,i]
        ufSE = u[k,j,i]

        wfNB = w[k-1,j+1,i] 
        wfSB = w[k-1,j,i]
        wfNF = w[k,j+1,i]
        wfSF = w[k,j,i]

        # convective terms:
        # d(uv)/dx +d(vv)/dy +d(wv)/dz
        conv = 0.0

        # d(uv)/dx
        # interp v in x 
        vfE = 0.5*( vcv +vcvE )
        vfW = 0.5*( vcvW +vcv )
        # interp u in y
        # compute required vcv half-lengths
        dyNh = 0.5*( y[j+2] -y[j+1] )  # length of north half of vcv
        dySh = 0.5*( y[j+1] -y[j] ) 
        ufW = ( ufNW*dySh +ufSW*dyNh ) / ( dySh +dyNh ) 
        ufE = ( ufNE*dySh +ufSE*dyNh ) / ( dySh +dyNh )

        conv += ( ufE*vfE -ufW*vfW ) / dx

        # d(v2)/dy
        # compute vcv length
        dy = 0.5*( y[j+2] +y[j+1] ) -0.5*( y[j+1] +y[j] )
        # interp v in y (F to C)
        vfN = 0.5*( vcv +vcvN )
        vfS = 0.5*( vcvS +vcv )

        conv += ( vfN**2 -vfS**2 ) / dy

        # d(wv)/dz
        # interp w in y
        wfB = ( wfNB*dySh +wfSB*dyNh ) / ( dySh +dyNh )
        wfF = ( wfNF*dySh +wfSF*dyNh ) / ( dySh +dyNh )
        # interp v in z
        vfB = 0.5*( vcv +vcvB )
        vfF = 0.5*( vcv +vcvF )

        conv += ( wfF*vfF -wfB*vfB ) / dz

        # diffusive terms:
        # d2(v)/dx2 +d2(v)/dy2 +d2(v)/dz2
        diff = 0.0

        # d2(v)/dx2
        dvdxE = ( vcvE -vcv ) / dx
        dvdxW = ( vcv -vcvW ) / dx
        diff += NU*( dvdxE -dvdxW ) / dx

        # d2(v)/dy2
        # compute required distances to N and S vcvs
        dyN = y[j+2] -y[j+1]
        dyS = y[j+1] -y[j]
        dvdyN = ( vcvN -vcv ) / dyN
        dvdyS = ( vcv -vcvS ) / dyS
        diff += NU*( dvdyN -dvdyS ) / dy

        # d2(v)/dz2
        dvdzF = ( vcvF -vcv ) / dz
        dvdzB = ( vcv -vcvB ) / dz
        diff += NU*( dvdzF -dvdzB ) / dz

        rhs_v[k,j,i] = -conv +diff

        
    # z-momentum (final terms land in wcv-centers)
    if (i>=1 and i < ncv_x-1 ) and (j>=1 and j < ncv_y-1 ) and (k >= 1 and k < nno_z-1):

        # pre-compute useful velocities
        # compute neighbouring CV quantities
        wcv  = w[k,j,i]
        wcvE = w[k,j,i+1]
        wcvW = w[k,j,i-1]
        wcvN = w[k,j+1,i]
        wcvS = w[k,j-1,i]
        wcvF = w[k+1,j,i]
        wcvB = w[k-1,j,i]
        # compute CV corner quantities
        ufEF = u[k+1,j,i]
        ufWF = u[k+1,j,i-1]
        ufEB = u[k,j,i]
        ufWB = u[k,j,i-1]

        vfNF = v[k+1,j,i]
        vfSF = v[k+1,j-1,i]
        vfNB = v[k,j,i]
        vfSB = v[k,j-1,i]

        # convective terms:
        # d(uw)/dx +d(vw)/dy +d(w2)/dz
        conv = 0.0

        # d(uw)/dx
        # interp u in x so it is at E and W faces
        wfE = 0.5*( wcv +wcvE )
        wfW = 0.5*( wcvW +wcv )
        # interp u in x
        ufE = 0.5*( ufEF +ufEB )
        ufW = 0.5*( ufWF +ufWB )

        conv += ( ufE*wfE -ufW*wfW ) / dx

        # d(vw)/dy
        # interp v in z
        vfN = 0.5*( vfNB +vfNF )
        vfS = 0.5*( vfSB +vfSF )
        # interp w in y
        # compute required wcv lengths
        dy = y[j+1] -y[j]
        dyN = y[j+2] -y[j+1]
        dyS = y[j] -y[j-1]
        # interp w in y so it is at N and S faces
        wfN = ( (dyN/2)*wcv +(dy/2)*wcvN ) / (dyN/2 +dy/2)
        wfS = ( (dy/2)*wcvS +(dyS/2)*wcv ) / (dy/2 +dyS/2)

        conv += ( vfN*wfN -vfS*wfS ) / dy

        # d(w2)/dz
        # interp w in z
        wfF = 0.5*( wcv +wcvF )
        wfB = 0.5*( wcvB +wcv )

        conv += ( wfF*wfF -wfB*wfB ) / dz


        # diffusive terms:
        # d2(w)/dx2 +d2(w)/dy2 +d2(w)/dz2
        diff = 0.0

        # d2(w)/dx2
        # compute dwdx (lands on E/W faces)
        dwdxE = ( wcvE -wcv ) / dx
        dwdxW = ( wcv -wcvW ) / dx
        # compute d2wdx2
        diff += NU*( dwdxE -dwdxW ) / dx

        # d2(w)dy2
        # compute dwdy (lands on N/S faces)
        PFN = (dyN/2)/(dy/2)
        # see Sundqvist and Veronis (1969) eq 1.3
        dwdyN = ( wcvN-(PFN**2)*wcv -(1 -(PFN**2))*wfN ) /\
                (dyN/2*(1+PFN))
        PFS = (dy/2)/(dyS/2)
        dwdyS = (wcv-(PFS**2)*wcvS -(1 -(PFS**2))*wfS) /\
                (dy/2*(1+PFS))

        diff += NU*( dwdyN -dwdyS ) / dy

        # d2(w)dz2
        # compute dwdz (lands on F/B faces)
        dwdzF = ( wcvF -wcv ) / dz
        dwdzB = ( wcv -wcvB ) / dz
        # compute d2wdz2
        diff += NU*( dwdzF -dwdzB ) / dz

        rhs_w[k,j,i] = -conv +diff


@cuda.jit
def gs_update(p, a_east, a_west, a_north, a_south, a_front, a_back, a_p, b, color_phase):
    k, j, i = cuda.grid(3)

    if (i>=1 and i< ncv_x-1) and (j>=1 and j< ncv_y-1) and (k>=1 and k< ncv_z-1):
        if (i + j + k) % 2 == color_phase:
            # assumes ghosts have been properly updated
            tmp = (b[k, j, i] - \
                   a_east[k, j, i] * p[k, j, i+1] - \
                   a_west[k, j, i] * p[k, j, i-1] - \
                   a_north[k, j, i] * p[k, j+1, i] - \
                   a_south[k, j, i] * p[k, j-1, i] - \
                   a_front[k, j, i] * p[k+1, j, i] - \
                   a_back[k, j, i] * p[k-1, j, i]) / a_p[k, j, i]
            p[k,j,i] = tmp


@cuda.jit
def populate_rhs( dt, dx, y, dz, p,b,ut,vt,wt, ncv_x, ncv_y, ncv_z, nno_x, nno_y, nno_z ):
    k, j, i = cuda.grid(3)

    # b = (1/dt)*div(u)
    if (i >= 1 and i < ncv_x-1) and (j >= 1 and j < ncv_y-1) and (k >= 1 and k < ncv_z-1):

        dy = y[j+1]-y[j]  
        # F to C 
        b[k,j,i] = (1/dt)*( (ut[k,j,i] -ut[k,j,i-1])/dx +\
                            (vt[k,j,i] -vt[k,j-1,i])/dy +\
                            (wt[k,j,i] -wt[k-1,j,i])/dz )


@cuda.jit
def correct_uvw( dt,dx,y,dz,u,v,w,p,ut,vt, wt, ncv_x, ncv_y, ncv_z, nno_x, nno_y, nno_z ):
    k, j, i = cuda.grid(3)
    # correct u
    if (i >= 1 and i < nno_x-1) and (j>=1 and j < ncv_y-1) and (k>=1 and k < ncv_z-1):
        u[k, j, i] = ut[k, j, i] - (dt/dx) * (p[k, j, i+1] - p[k, j, i])
    # correct v
    if (i>=1 and i< ncv_x-1) and (j >= 1 and j < nno_y-1) and (k>=1 and k < ncv_z-1):
        # need pressure at y-faces
        dyN = 0.5*(y[j+2]-y[j+1])
        dyS = 0.5*(y[j+1]-y[j])
        pf = ( dyS*p[k,j+1,i] + dyN*p[k,j,i] ) / (dyS+dyN)
        v[k, j, i] = vt[k, j, i] - dt * ( (p[k,j+1,i] -pf)/dyN**2 +\
                     (pf -p[k,j,i])/dyS**2 ) * (dyS*dyN)/(dyS+dyN)
    # correct w
    if (i>=1 and i < ncv_x-1) and (j>=1 and j < ncv_y-1) and (k >= 1 and k < nno_z-1):
        w[k, j, i] = wt[k, j, i] - (dt/dz) * (p[k+1, j, i] - p[k, j, i])


@cuda.jit
def apply_BCs( u, v, w, p, ncv_x, ncv_y, ncv_z, nno_x, nno_y, nno_z ):

    # periodicity in x (u at x-faces)
    j, k = cuda.grid(2)
    if (j < ncv_y) and (k < ncv_z):
        u[k, j, 0] = u[k, j, -2]
        u[k, j, -1] = u[k, j, 1]

    # no-slip in y
    k,i = cuda.grid(2)
    if (i < nno_x) and (k < ncv_z):
        u[k,0,i] = -u[k,1,i]
        u[k,-1,i] = -u[k,-2,i]
        
    # periodicity in z
    if (i < nno_x) and (j < ncv_y):
        u[0,j,i] = u[-3,j,i]
        u[-2,j,i] = u[1,j,i]
        u[-1,j,i] = u[2,j,i]

    # periodicity in x (v at x-faces)
    j, k = cuda.grid(2)
    if (j < nno_y) and (k < ncv_z):
        v[k, j, 0] = v[k, j, -3]
        v[k, j, -2] = v[k, j, 1]
        v[k, j, -1] = v[k, j, 2]

    # no-slip in y
    k,i = cuda.grid(2)
    if (i < ncv_x) and (k < ncv_z):
        v[k,0,i] = 0.0
        v[k,-1,i] = 0.0
        
    # periodicity in z
    i,j = cuda.grid(2)
    if (i < ncv_x) and (j < nno_y):
        v[0,j,i] = v[-3,j,i]
        v[-2,j,i] = v[1,j,i]
        v[-1,j,i] = v[2,j,i]

    # periodicity in x (w at z-faces)
    j, k = cuda.grid(2)
    if (j < ncv_y) and (k < nno_z):
        w[k, j, 0] = w[k, j, -3]
        w[k, j, -2] = w[k, j, 1]
        w[k, j, -1] = w[k, j, 2]

    # no-slip in y
    k,i = cuda.grid(2)
    if (i < ncv_x) and (k < nno_z):
        w[k,0,i] = -w[k,1,i]
        w[k,-1,i] = -w[k,-2,i]
        
    # periodicity in z
    i,j = cuda.grid(2)
    if (i < ncv_x) and (j < ncv_y):
        w[0,j,i] = w[-2,j,i]
        w[-1,j,i] = w[1,j,i]

    # periodicity in x (p is in cell-centers)
    j, k = cuda.grid(2)
    if (j < ncv_y) and (k < ncv_z):
        p[k, j, 0] = p[k, j, -3]
        p[k, j, -2] = p[k, j, 1]
        p[k, j, -1] = p[k, j, 2]

    # zero-flux Neumann in y
    k,i = cuda.grid(2)
    if (i < ncv_x) and (k < ncv_z):
        p[k,0,i] = p[k,1,i]
        p[k,-1,i] = p[k,-2,i]
        
    # periodicity in z
    i,j = cuda.grid(2)
    if (i < ncv_x) and (j < ncv_y):
        p[0,j,i] = p[-3,j,i]
        p[-2,j,i] = p[1,j,i]
        p[-1,j,i] = p[2,j,i]


nno_x = N
nno_y = N
nno_z = N

ncv_x = nno_x +1
ncv_y = nno_y +1
ncv_z = nno_z +1

u = np.zeros((ncv_z,ncv_y,nno_x))
v = np.zeros((ncv_z,nno_y,ncv_x))
w = np.zeros((nno_z,ncv_y,ncv_x))
p = np.zeros((ncv_z,ncv_y,ncv_x))
ut = np.zeros_like(u)
vt = np.zeros_like(v)
wt = np.zeros_like(w)

# grid points (nodes)
x = np.linspace(0,2*np.pi,nno_x)
y = np.linspace(-1.0, 1.0, nno_y)  
#y = np.tanh(1.2 * y) / np.tanh(1.2) 
# remember that y and v are offset by 1 in indexing because of below:
y = np.insert(y, 0, y[0]-(y[1]-y[0]))
y = np.append(y, y[-1]+(y[-1]-y[-2]))
z = np.linspace(0.0,np.pi,nno_z)

# equal spacing
dx = x[1]-x[0]
dz = z[1]-z[0]

dx_min = min( dx, np.min(y[1:] -y[:-1]), dz)
#dt = min( 0.1*dx_min**2/NU, CFL*dx_min/25.0 )

# initialize the matrix of coefficient
a_east  = np.zeros((ncv_z,ncv_y,ncv_x))
a_west  = np.zeros((ncv_z,ncv_y,ncv_x))
a_north = np.zeros((ncv_z,ncv_y,ncv_x))
a_south = np.zeros((ncv_z,ncv_y,ncv_x))
a_front = np.zeros((ncv_z,ncv_y,ncv_x))
a_back = np.zeros((ncv_z,ncv_y,ncv_x))
a_p     = np.zeros((ncv_z,ncv_y,ncv_x))
b       = np.zeros((ncv_z,ncv_y,ncv_x))

# build the matrix for the pressure equation
for k in range(1,ncv_z-1):
    for j in range(1,ncv_y-1):
        for i in range(1,ncv_x-1):
            a_west[k,j,i] = 1/dx**2
            a_east[k,j,i] = 1/dx**2

            dyS = 0.5*(y[j+1]+y[j]) -0.5*(y[j]+y[j-1])
            dyN = 0.5*(y[j+2]+y[j+1]) -0.5*(y[j+1]+y[j]) 
            a_south[k,j,i] = 2/(dyS*(dyN +dyS))
            a_north[k,j,i] = 2/(dyN*(dyN +dyS))

            a_back[k,j,i] = 1/dz**2
            a_front[k,j,i] = 1/dz**2

a_p = -( a_north +a_south +a_east +a_west +a_front +a_back )


# IC
#u = np.random.random(u.shape)*1e-0
#v = np.random.random(v.shape)*1e-0
##w = np.random.random(w.shape)*1e-0

u = np.load('lam-sol/u749000.npy')
v = np.load('lam-sol/v749000.npy')
w = np.load('lam-sol/w749000.npy')
p = np.load('lam-sol/p749000.npy')

#u = u*(1 +np.random.random(u.shape)*1e-2)
#v = v*(1 +np.random.random(v.shape)*1e-2)
#w = w*(1 +np.random.random(w.shape)*1e-5)
amp = 0#0.25 
u += amp * (np.random.random(u.shape) - 0.5)
#v += amp * (np.random.random(v.shape) - 0.5)
#w += amp * (np.random.random(w.shape) - 0.5)

print(f'u_max: {np.max(u)}')
print(f'v_max: {np.max(v)}')
print(f'w_max: {np.max(w)}')

## cuda stuff begins:
# move all arrays to device
y_ = cuda.to_device(y)
u_ = cuda.to_device(u)
v_ = cuda.to_device(v)
w_ = cuda.to_device(w)
p_ = cuda.to_device(p)
ut_ = cuda.to_device(ut)
vt_ = cuda.to_device(vt)
wt_ = cuda.to_device(wt)
a_east_ = cuda.to_device(a_east)
a_west_ = cuda.to_device(a_west)
a_north_ = cuda.to_device(a_north)
a_south_ = cuda.to_device(a_south)
a_front_ = cuda.to_device(a_front)
a_back_ = cuda.to_device(a_back)
a_p_ = cuda.to_device(a_p)
b_ = cuda.to_device(b)

# allocate RK3 stuff
rhs_u1_ = cuda.device_array_like(u_)
rhs_u2_ = cuda.device_array_like(u_)
rhs_u3_ = cuda.device_array_like(u_)
rhs_v1_ = cuda.device_array_like(v_)
rhs_v2_ = cuda.device_array_like(v_)
rhs_v3_ = cuda.device_array_like(v_)
rhs_w1_ = cuda.device_array_like(w_)
rhs_w2_ = cuda.device_array_like(w_)
rhs_w3_ = cuda.device_array_like(w_)

# allocate device arrays for RK-3 integrator
u_buf_ = cuda.device_array_like(u_)
v_buf_ = cuda.device_array_like(v_)
w_buf_ = cuda.device_array_like(w_)

# RK-3 coefficients
rk_coef11 = 8.0/15
rk_coef21 = 1.0/4
rk_coef22 = 5.0/12
rk_coef31 = 1.0/4
rk_coef32 = 0.0
rk_coef33 = 3.0/4


tim = 0.0

BPG = ( int(np.ceil(ncv_x / TPB[0])), \
        int(np.ceil(ncv_y / TPB[1])), \
        int(np.ceil(ncv_z / TPB[2])) )


@vectorize([float64(float64, float64, float64)])
def rk_update_(dt, rk_coef, rhs):
    return dt*rk_coef * rhs

for istep in range(timesteps):

    # RK3 step 1
    u_buf_[:] = u_
    v_buf_[:] = v_
    w_buf_[:] = w_

    solve_uvw[BPG, TPB]   ( dt,dx, y_,dz, u_,v_,w_,rhs_u1_,rhs_v1_,rhs_w1_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    ut_ = u_ +rk_update_(dt, rk_coef11, rhs_u1_)
    vt_ = v_ +rk_update_(dt, rk_coef11, rhs_v1_)
    wt_ = w_ +rk_update_(dt, rk_coef11, rhs_w1_)

    apply_BCs[BPG, TPB]   ( ut_, vt_, wt_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    populate_rhs[BPG, TPB]( dt,dx, y_, dz, p_,b_,ut_,vt_,wt_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    for _ in range(NITER):
        # update red nodes
        gs_update[BPG, TPB]( p_, a_east_, a_west_, a_north_, a_south_, \
                                      a_front_, a_back_, a_p_, b_, 0)
        # update black nodes
        cuda.synchronize()
        gs_update[BPG, TPB]( p_, a_east_, a_west_, a_north_, a_south_, \
                                      a_front_, a_back_, a_p_, b_, 1)
        cuda.synchronize()

    apply_BCs[BPG, TPB]   ( ut_, vt_, wt_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    cuda.synchronize()
    correct_uvw[BPG, TPB] ( dt,dx, y_,dz, u_,v_,w_,p_,ut_,vt_, wt_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    apply_BCs[BPG, TPB]   ( u_, v_, w_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    cuda.synchronize()


    # RK3 step 2

    solve_uvw[BPG, TPB]   ( dt,dx, y_,dz, u_,v_,w_,rhs_u2_,rhs_v2_,rhs_w2_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    ut_ = u_buf_ +rk_update_( dt, rk_coef21, rhs_u1_ ) +rk_update_( dt, rk_coef21, rhs_u2_ )
    vt_ = v_buf_ +rk_update_( dt, rk_coef21, rhs_v1_ ) +rk_update_( dt, rk_coef21, rhs_v2_ )
    wt_ = w_buf_ +rk_update_( dt, rk_coef21, rhs_w1_ ) +rk_update_( dt, rk_coef21, rhs_w2_ )

    apply_BCs[BPG, TPB]   ( ut_, vt_, wt_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    populate_rhs[BPG, TPB]( dt,dx, y_, dz, p_,b_,ut_,vt_,wt_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    for _ in range(NITER):
        # update red nodes
        gs_update[BPG, TPB]( p_, a_east_, a_west_, a_north_, a_south_, \
                                      a_front_, a_back_, a_p_, b_, 0)
        # update black nodes
        cuda.synchronize()
        gs_update[BPG, TPB]( p_, a_east_, a_west_, a_north_, a_south_, \
                                      a_front_, a_back_, a_p_, b_, 1)
        cuda.synchronize()

    apply_BCs[BPG, TPB]   ( ut_, vt_, wt_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    cuda.synchronize()
    correct_uvw[BPG, TPB] ( dt,dx, y_,dz, u_,v_,w_,p_,ut_,vt_, wt_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    apply_BCs[BPG, TPB]   ( u_, v_, w_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    cuda.synchronize()

    # RK3 step 3

    solve_uvw[BPG, TPB]   ( dt,dx, y_,dz, u_,v_,w_,rhs_u3_,rhs_v3_,rhs_w3_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    ut_ = u_buf_ +rk_update_( dt, rk_coef31, rhs_u1_ ) +rk_update_( dt, rk_coef32, rhs_u2_ ) +\
                  rk_update_( dt, rk_coef33, rhs_u3_ )
    vt_ = v_buf_ +rk_update_( dt, rk_coef31, rhs_v1_ ) +rk_update_( dt, rk_coef32, rhs_v2_ ) +\
                  rk_update_( dt, rk_coef33, rhs_v3_ )
    wt_ = w_buf_ +rk_update_( dt, rk_coef31, rhs_w1_ ) +rk_update_( dt, rk_coef32, rhs_w2_ ) +\
                  rk_update_( dt, rk_coef33, rhs_w3_ )

    apply_BCs[BPG, TPB]   ( ut_, vt_, wt_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    populate_rhs[BPG, TPB]( dt,dx, y_, dz, p_,b_,ut_,vt_,wt_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    for _ in range(NITER):
        # update red nodes
        gs_update[BPG, TPB]( p_, a_east_, a_west_, a_north_, a_south_, \
                                      a_front_, a_back_, a_p_, b_, 0)
        # update black nodes
        cuda.synchronize()
        gs_update[BPG, TPB]( p_, a_east_, a_west_, a_north_, a_south_, \
                                      a_front_, a_back_, a_p_, b_, 1)
        cuda.synchronize()

    apply_BCs[BPG, TPB]   ( ut_, vt_, wt_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )

    cuda.synchronize()
    correct_uvw[BPG, TPB] ( dt,dx, y_,dz, u_,v_,w_,p_,ut_,vt_, wt_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    apply_BCs[BPG, TPB]   ( u_, v_, w_, p_, ncv_x, ncv_y, ncv_z, \
                           nno_x, nno_y, nno_z )
    cuda.synchronize()


    print(f'Time step size: {dt:.8f}')
    print(f'Step count: {istep} of {timesteps}')
    print(f'Simulation time: {tim:.8f}')

    tim = tim + dt

    if ( istep % 1000 == 0 ):
        u = u_.copy_to_host()
        v = v_.copy_to_host()
        w = w_.copy_to_host()
        p = p_.copy_to_host()

        np.save(f'npys/u{istep}.npy', u)
        np.save(f'npys/v{istep}.npy', v)
        np.save(f'npys/w{istep}.npy', w)
        np.save(f'npys/p{istep}.npy', p)


np.save(f'npys/y.npy', y)
np.save(f'npys/x.npy', x)
np.save(f'npys/z.npy', z)

print(u[int(u.shape[0]//2), :,:])
print(v[:, int(v.shape[1]//2),:])
print(w[:,:, int(w.shape[1]//2)])
print(p[int(p.shape[1]//2),:,:])

