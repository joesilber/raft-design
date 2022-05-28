from scipy import optimize
import numpy as np

class FitCircle(object):
    """C.f. http://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle"""
    
    def fit(self, xy):
        """Return a best-fit circle for (x,y) data.
        
        Input:    xy ... cartesian points of the form [[x1,y1], [x2,y2], ...]
        
        Returns:  xy_ctr ... best-fit circle center, in the form [x_center,y_center]
                  radius ... best-fit circle radius
        """
        x = [pt[0] for pt in xy]
        y = [pt[1] for pt in xy]
        
        def calc_R(xc, yc):
            """calculate the distance of each 2D points from the center (xc, yc) """
            return ((x-xc)**2 + (y-yc)**2)**0.5   
            
        def func(c):
            """calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()
        
        xm = np.mean(x)
        ym = np.mean(y)
        center_estimate = xm, ym
        center, ier = optimize.leastsq(func, center_estimate)
        xc, yc = center
        Ri = calc_R(*center)
        R = Ri.mean()
        #residu = sum((Ri - R)**2) # residual
        xy_ctr = [xc,yc]
        return  xy_ctr, R

if __name__ == '__main__':
    # xy = [[1.0,0],[0,1],[-1,0]]
    # xy = np.array(xy) * 7.924 + [3,2]
    xy = [(275.69630069404354, 120.49879624120292), (275.3969906269045, 121.10874991791466), (274.9718340417679, 121.63872339096775), (274.4413083936607, 122.06319075216918), (273.83096618601195, 122.36170774261942), (273.17020424706124, 122.51989644025309)]
    xy_ctr, radius = FitCircle().fit(xy)
    print('xy_ctr: (' + str(xy_ctr[0]) + ', ' + str(xy_ctr[1]) + ')')
    print('radius: ' + str(radius))
