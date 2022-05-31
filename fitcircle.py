from scipy import optimize
import numpy as np

class FitCircle(object):
    """C.f. http://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle"""
    
    def fit(self, xy, weight_as_sphere=False):
        """Return a best-fit circle for (x,y) data.
        
        Input:    xy ... cartesian points of the form [[x1,y1], [x2,y2], ...]
                  weight_as_sphere ... argue True to weight the fit as if the x,y data
                                       revolved about a z-axis, forming a sphere
        
        Returns:  xy_ctr ... best-fit circle center, in the form [x_center,y_center]
                  radius ... best-fit circle radius
        """
        xynp = np.transpose(xy)
        x = xynp[0]
        y = xynp[1]
        r = np.hypot(x, y)
        
        def calc_R(xc, yc):
            """calculate the distance of each 2D points from the center (xc, yc)"""
            return ((x-xc)**2 + (y-yc)**2)**0.5   
            
        def errfunc(center):
            """calculate the algebraic distance between the data points and the mean circle centered at (xc, yc)"""
            Rtest = calc_R(*center)
            errors = Rtest - Rtest.mean()
            if weight_as_sphere:
                errors *= r / max(r)
            scalar_error = np.sum(np.power(errors, 2))
            return scalar_error
        
        xm = np.mean(x)
        ym = np.mean(y)
        #center_estimate = xm, ym
        #center, ier = optimize.leastsq(errfunc, center_estimate)
        optim_tol = max(r) / 1e6
        result = optimize.minimize(fun=errfunc, x0=(xm, ym), tol=optim_tol)#options={'maxiter': 1000})
        result2 = optimize.least_squares(fun=errfunc, x0=(xm, ym), xtol=optim_tol)#options={'maxiter': 1000})


        xc, yc = result.x.tolist()
        Rtest = calc_R(xc, yc)
        Rmean = np.mean(Rtest)
        xy_ctr = [xc, yc]
        return  xy_ctr, Rmean

if __name__ == '__main__':
    # xy = [[1.0,0],[0,1],[-1,0]]
    # xy = np.array(xy) * 7.924 + [3,2]
    xy = [(275.69630069404354, 120.49879624120292), (275.3969906269045, 121.10874991791466), (274.9718340417679, 121.63872339096775), (274.4413083936607, 122.06319075216918), (273.83096618601195, 122.36170774261942), (273.17020424706124, 122.51989644025309)]
    xy_ctr, radius = FitCircle().fit(xy)
    print('xy_ctr: (' + str(xy_ctr[0]) + ', ' + str(xy_ctr[1]) + ')')
    print('radius: ' + str(radius))

    import matplotlib.pyplot as plt
    xynp = np.transpose(xy)
    plt.plot(xynp[0], xynp[1], 'o')
    plt.show()