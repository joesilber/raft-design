import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import optimize
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from astropy.table import Table

_raft_id_counter = 0

class Raft:
    '''Represents a single triangular raft.'''
    
    def __init__(self, x0=0., y0=0., spin0=0.,
                 focus_offset=0., tilt_offset=0.,
                 outer_profile=None, instr_profile=None,
                 r2nut=None, r2z=None, sphR=math.inf,
                 robot_pitch=6.2, robot_max_extent=4.4,
                 ):
        '''
        x0 ... [mm] x location of center of front triangle, *not* including tilt and focus offset compensations
        y0 ... [mm] y location of center of front triangle, *not* including tilt and focus offset compensations
        spin0 ... [deg] rotation of triangle, *not* including precession compensation
        focus_offset ... [mm], use this parameter to adjust focus of the overall raft (i.e. when optimizing)
        tilt_offset ... [mm], use this parameter to adjust nutation of the overall raft (i.e. when optimizing)
        outer_profile ... RaftProfile instance, defining outer geometry
        instr_profile ... RaftProfile instance, defining instrumented geometry
        r2nut ... [mm --> mm] function for converting focal plane radius to nutation angle of raft
        r2z ... [mm --> mm] function for converting focal plane radius to z position of raft
        sphR ... [mm] spherical radius which approximates focal surface; sign convention is:
                 sphR > 0 means concave, sphR < 0 means convex, sphR ~ infinity means flat
        robot_pitch ... [mm] center-to-center distance between robots within the raft
        robot_max_extent ... [mm] local to a robot, max radius of any mechanical part at full extension
        '''
        global _raft_id_counter
        self.id = _raft_id_counter
        _raft_id_counter += 1
        self.x0 = x0
        self.y0 = y0
        self.spin0 = spin0
        self.focus_offset = focus_offset
        self.tilt_offset = tilt_offset
        self.neighbors = []
        self.outer_profile = outer_profile if outer_profile else RaftProfile()
        self.instr_profile = instr_profile if instr_profile else RaftProfile(tri_base=79., chamfer=7.9)
        self.r2nut = r2nut if r2nut else lambda x: np.zeros(np.shape(x))
        self.r2z = r2z if r2z else lambda x: np.zeros(np.shape(x))
        self.sphR = sphR
        self.robot_pitch = robot_pitch
        self.robot_max_extent = robot_max_extent

    @property
    def x(self):
        '''x position [mm] of center of raft at front, including corrections
        for focus_offset and tilt_offset parameters'''
        return self.r * math.cos(math.radians(self.precession0))

    @property
    def y(self):
        '''x position [mm] of center of raft at front, including corrections
        for focus_offset and tilt_offset parameters'''
        return self.r * math.sin(math.radians(self.precession0))

    @property
    def r(self):
        '''radial position [mm] of center of raft at front, including corrections
        for focus_offset and tilt_offset parameters''' 
        offset_r_correction = self.focus_offset * math.sin(math.radians(self.nutation))
        return self.r0 + offset_r_correction

    @property
    def r0(self):
        '''radial position [mm] of center of raft at front, *not* including
        corrections for focus_offset and tilt_offset parameters'''
        return math.hypot(self.x0, self.y0)
    
    @property
    def z(self):
        '''z position [mm] of center of raft at front, including corrections
        for focus_offset and tilt_offset parameters'''
        z0 = float(self.r2z(self.r))
        offset_z_correction = self.focus_offset * math.cos(math.radians(self.nutation))
        return z0 + offset_z_correction

    @property
    def precession(self):
        '''angular position [deg] about the z-axis, including corrections for
        focus_offset and tilt_offset parameters'''
        return math.degrees(math.atan2(self.y, self.x))

    @property
    def precession0(self):
        '''angular position [deg] about the z-axis, *not* including
        corrections for focus_offset and tilt_offset parameters'''
        return math.degrees(math.atan2(self.y0, self.x0))

    @property
    def nutation(self):
        '''angle [deg] w.r.t. z-axis, where if tilt_offset parameter is zero,
        then it would match chief ray at center of raft'''
        nut0 = float(self.r2nut(self.r0))
        return nut0 + self.tilt_offset
    
    @property
    def spin(self):
        '''rotation [deg] about raft's local z-axis, *including* compensation for
        precession (since raft orientation is defined by a 3-2-3 Euler rotation)'''
        return float(self.spin0 - self.precession)

    @property
    def z_vector(self):
        '''unit vector pointing in direction of raft z-axis'''
        return Raft.pn2zvec(self.precession, self.nutation)

    @property
    def n_robots(self):
        '''number of robots on this raft'''
        return len(self.instr_profile.generate_robot_pattern(pitch=self.robot_pitch))

    def front_poly(self, instr=False):
        '''Nx3 list of polygon vertices giving raft profile at front (i.e. at focal
        surface). Set arg instr=True to use the smaller instrumented area profile'''
        profile = self.instr_profile if instr else self.outer_profile
        poly = profile.polygon3D
        placed = self.place_poly(poly)
        return placed.tolist()

    @property
    def rear_poly(self):
        '''Nx3 list of polygon vertices giving raft profile at rear (i.e. at
        connectors bulkhead, etc)'''
        poly = np.array(self.outer_profile.polygon3D) + [0, 0, -self.outer_profile.RL]
        placed = self.place_poly(poly)
        return placed.tolist()

    @property
    def poly3d(self):
        '''Nx3 list of polygon vertices, intended for 3D plotting, includes front
        and rear outer closed polygons'''
        front = self.front_poly(instr=False)
        rear = self.rear_poly
        poly3d = front + [front[0]]
        for i in range(len(rear) - 1):
            poly3d += [rear[i], rear[i+1], front[i+1]]
        poly3d += [rear[i+1], rear[0]]
        return poly3d
    
    @property
    def poly3d_instr(self):
        '''Nx3 list of polygon vertices, intended for 3D plotting, includes front
        instrumented area closed polygon'''
        poly3d = self.front_poly(instr=True)
        poly3d += [poly3d[0]]
        return poly3d
    
    def generate_robots_table(self, global_coords=True):
        '''Returns an astropy table describing the individual robots on the raft.

        Boolean argument global_coords controls whether to return coordinates and
        angles local to the raft or global to the focal plane.

        Returned table includes columns:
            idx ... integer id, locally unique within this raft
            x, y, z ... cartesian coordinates of individual robot centers
            precession, nutation, spin ... angle of robot central axes
            intersects perimeter ... whether each robot's patrol disk intersects outline of raft
        '''
        points3D = self.generate_local_robot_centers_no_offsets()
        points2D = np.transpose(points3D)[:2]
        n_pts = len(points2D[0])
        if global_coords:
            points3D = self.place_poly(points3D)
            # 2022-11-30 - JHS - Current design assumption is we will  
            # keep all robot center axes parallel to the raft.
            angles = np.ones_like(points3D) * [self.precession, self.nutation, self.spin]
        else:
            angles = np.zeros_like(points3D)
        intersects_perimeter = [self.instr_profile.circle_intersects(points2D[0,i], points2D[1,i], self.robot_max_extent) for i in range(n_pts)]
        data = {'idx': np.arange(n_pts),
                'x': points3D[:,0],
                'y': points3D[:,1],
                'z': points3D[:,2],
                'precession': angles[:,0],
                'nutation': angles[:,1],
                'spin': angles[:,2],
                'intersects perimeter': intersects_perimeter,
                }
        data['r'] = np.hypot(data['x'], data['y'])
        table = Table(data)
        return table
    
    def generate_local_robot_centers_no_offsets(self):
        '''generate 3D pattern of center positions of the robots. pattern is centered
        at raft origin, with robot centers placed on nominal sphere. defocus and tilt
        offset adjustments are *not* applied'''
        points2D = self.instr_profile.generate_robot_pattern(pitch=self.robot_pitch)
        points2D = np.transpose(points2D)
        local_r = np.hypot(points2D[0], points2D[1])
        if np.isneginf(self.sphR) or np.isposinf(self.sphR):
            dz = np.zeros(np.shape(local_r))
        else:
            sign = np.sign(self.sphR)
            dz = self.sphR - sign*(self.sphR**2 - local_r**2)**0.5
        points3D = np.transpose(np.append(points2D, [dz], axis=0))
        return points3D.tolist()

    def max_front_vertex_radius(self, instr=False):
        '''maximum distance from the z-axis of any point in the 3d raft polygon
        set arg instr True to use the smaller instrumented area profile'''
        return max(self._front_vertex_radii(instr))

    def min_front_vertex_radius(self, instr=False):
        '''minimum distance from the z-axis of any point in the 3d raft polygon
        set arg instr True to use the smaller instrumented area profile'''
        return min(self._front_vertex_radii(instr))

    def _front_vertex_radii(self, instr=False):
        '''return distances of all points in the 3d raft polygon from the z-axis
        set arg instr True to use the smaller instrumented area profile'''
        all_points = np.transpose(self.front_poly(instr))
        return np.hypot(all_points[0], all_points[1])

    def front_gap(self, other_raft):
        '''Returns min distance and perpendicular unit vector from closest segment on this
        raft's front polygon toward corresponding closest point on "other" raft.'''
        return Raft.poly_gap(other_raft.front_poly(), self.front_poly())

    def rear_gap(self, other_raft):
        '''Returns min distance and perpendicular unit vector from closest segment on this
        raft's front polygon toward corresponding closest point on "other" raft.'''
        return Raft.poly_gap(other_raft.rear_poly, self.rear_poly)

    def place_poly(self, poly):
        '''Transform a polygon (N x 3) from the origin to the raft's center position on the
        focal surface. The polygon is first rotated such that a vector (0, 0, 1) becomes
        its final orientation (including parameter tilt_offset) when placed at the corresponding
        radius, and such that a point (0, 0, 0) will land on the nominal focal surface (if the
        parameter focus_offset = 0) or at the distance focus_offset from it.'''
        rot = Rotation.from_euler('ZYZ', (self.precession, self.nutation, self.spin), degrees=True)
        rotated = rot.apply(poly)
        translated = rotated + [self.x, self.y, self.z]
        return translated

    @staticmethod
    def poly_gap(poly1, poly2):
        '''Returns a magnitude and direction unit vector for closest distance between two polygons.
        This is calculated from segments defined by poly2 to vertices defined by poly1. If the two
        polygons overlap, returns a default value of (0, np.array([1,0,0])). The input polygons should
        be Nx2 arrays of (x, y) or Nx3 arrays of (x, y, z) vertices. The returned unit vector points
        perpendicularly from the nearest poly2 segment toward the corresponding nearest poly1 point.'''
        if Raft.polygons_collide(poly1, poly2):
            return 0, np.array([1, 0, 0])
        test_pts = [np.array(pt) for pt in poly1]
        segment_pts = [(poly2[i], poly2[i+1]) for i in range(len(poly2) - 1)]
        segment_pts += [(poly2[-1], poly2[0])]  # close the polygon with last segment
        min_dist = math.inf
        for seg in segment_pts:
            s0 = np.array(seg[0])
            s1 = np.array(seg[1])
            seg_vec = s1 - s0
            seg_mag = np.sqrt(seg_vec.dot(seg_vec))
            seg_unit = seg_vec / seg_mag
            for pt in test_pts:
                s2_mag = np.dot(pt - s0, seg_unit)
                if s2_mag < 0:
                    gap_vec_start = s0
                elif s2_mag > seg_mag:
                    gap_vec_start = s1
                else:
                    s2 = s2_mag * seg_unit
                    gap_vec_start = s2
                gap_vec = pt - gap_vec_start
                gap_mag = np.sqrt(gap_vec.dot(gap_vec))
                if gap_mag < min_dist:
                    min_dist = gap_mag
                    min_vec_unit = gap_vec / gap_mag
        return min_dist, min_vec_unit

    @staticmethod
    def polygons_collide(poly1, poly2):
        """Check whether two closed polygons collide.

        poly1 ... Nx2 array of the 1st polygon's vertices
        poly2 ... Nx2 array of the 2nd polygon's vertices

        Returns True if the polygons intersect, False if they do not.

        The algorithm is by detecting intersection of line segments, therefore the case of
        a small polygon completely enclosed by a larger polygon will return False. Not checking
        for this condition admittedly breaks some conceptual logic, but this case is not
        anticipated to occur given the DESI petal geometry, and speed is at a premium.
        """
        for i in range(len(poly1) - 1):
            for j in range(len(poly2) - 1):
                if Raft.segments_intersect(poly1[i], poly1[i+1], poly2[j], poly2[j+1]):
                    return True
        return False

    @staticmethod
    def segments_intersect(A1, A2, B1, B2):
        """Checks whether two 2d line segments intersect. The endpoints for segments
        A and B are each a pair of (x, y) coordinates.
        """
        dx_A = A2[0] - A1[0]
        dy_A = A2[1] - A1[1]
        dx_B = B2[0] - B1[0]
        dy_B = B2[1] - B1[1]
        delta = dx_B * dy_A - dy_B * dx_A
        if delta == 0.0:
            return False  # parallel segments
        s = (dx_A * (B1[1] - A1[1]) + dy_A * (A1[0] - B1[0])) / delta
        t = (dx_B * (A1[1] - B1[1]) + dy_B * (B1[0] - A1[0])) / (-delta)
        return (0 <= s <= 1) and (0 <= t <= 1)

    @staticmethod
    def pn2zvec(precession, nutation):
        '''returns a unit vector made by rotating (0, 0, 1) by angles precession
        and nutation (units degrees)'''
        p = math.radians(precession)
        n = math.radians(nutation)
        return [math.cos(p) * math.sin(n),
                math.sin(p) * math.sin(n),
                math.cos(n)]
    
class RaftProfile:
    '''Basic 2D profile geometry of raft.'''

    def __init__(self, tri_base=80., length=657., chamfer=2.5):
        '''
        tri_base ... base length of triangular outline of raft
        length ... length of raft
        chamfer ... trim-depth of the three raft corners, measured from triangle tip towards center
        '''
        self.RB = tri_base
        self.RL = length
        self.RC = chamfer

    @property
    def h1(self):
        '''height from base of triangle to opposite tip '''
        return self.RB * 3**0.5 / 2

    @property
    def h2(self):
        '''height from base of triangle to center'''
        return self.RB / 3**0.5 / 2
    
    @property
    def h3(self):
        '''height from center of triangle to tip'''
        return self.RB / 3**0.5

    @property
    def CB(self):
        '''chamfer base length'''
        return self.RC * 2 / 3**0.5
    
    @property
    def polygon2D(self):
        '''Nx2 list of (x, y) vertices representing the profile geometry'''
        poly_x = [self.RB/2 - self.CB,  self.RB/2 - self.CB/2,          self.CB/2,         -self.CB/2,  -self.RB/2 + self.CB/2,  -self.RB/2 + self.CB]
        poly_y = [           -self.h2,        self.RC-self.h2,  self.h3 - self.RC,  self.h3 - self.RC,       self.RC - self.h2,              -self.h2]
        return np.transpose([poly_x, poly_y]).tolist()
    
    @property
    def polygon3D(self):
        '''Same as polygon2D but with a third column of zeros added for the user's convenience'''
        poly2D = self.polygon2D
        poly3D = np.column_stack((poly2D, [0]*np.shape(poly2D)[0]))
        return poly3D.tolist()

    def generate_robot_pattern(self, pitch=6.2, flip='optimal'):
        '''Produce 2D pattern of robots that fit within the polygon.
          INPUTS:   pitch ... [mm] center-to-center distance between robots within the raft
                    flip ... valid arguments:
                                'optimal' ... choose the best option automatically
                                'default' ... pattern based on nominal 75-robot raft design circa Dec 2022
                                'vertical' ... mirror the default pattern vertically
                                'horizontal' ... mirror the default pattern horizontally
          OUTPUTS:  Nx2 list of (x, y) positions
        '''
        if flip == 'optimal':
            patterns = {key: self.generate_robot_pattern(pitch=pitch, flip=key) for key in ['default', 'horizontal', 'vertical']}
            best_pattern_dist = -np.inf
            best_pattern_key = ''
            profile_polygon = self.polygon2D
            for key, pattern in patterns.items():
                min_dist_robot_ctr_to_profile = np.inf
                np_pattern = np.array(pattern)
                for i in range(len(profile_polygon)):
                    vertex1 = profile_polygon[i]
                    vertex2 = profile_polygon[i+1] if i < (len(profile_polygon) - 1) else profile_polygon[0]
                    numerator = np.abs((vertex2[0] - vertex1[0])*(vertex1[1] - np_pattern[:,1]) - (vertex1[0] - np_pattern[:,0])*(vertex2[1] - vertex1[1]))
                    denominator = ((vertex2[0] - vertex1[0])**2 + (vertex2[1] - vertex1[1])**2)**0.5
                    point_distances_to_segments = numerator / denominator
                    this_min_dist = min(point_distances_to_segments)
                    min_dist_robot_ctr_to_profile = min(min_dist_robot_ctr_to_profile, this_min_dist)
                if min_dist_robot_ctr_to_profile > best_pattern_dist:
                    best_pattern_key = key
                    best_pattern_dist = min_dist_robot_ctr_to_profile
            return patterns[best_pattern_key]

        # square-ish local robot pattern
        overwidth = self.RB * 2/3
        overwidth_count = math.ceil(overwidth / pitch)
        pattern_row_x = pitch * np.arange(-overwidth_count, overwidth_count)
        pattern_row_y = np.zeros(np.shape(pattern_row_x))
        offset_x = 0.  # overall offset of the pattern
        offset_y = -pitch / 3**0.5  # overall offset of the pattern
        step_x = pitch / 2  # row-to-row x pitch
        step_y = pitch * 3**0.5 / 2  # row-to-row y pitch
        pattern = np.array([[],[]])
        for i in range(-overwidth_count, overwidth_count):
            new_row_x = pattern_row_x + offset_x + step_x * (i % 2)
            new_row_y = pattern_row_y + offset_y + step_y * i
            pattern = np.append(pattern, [new_row_x, new_row_y], axis=1)
        pattern = np.transpose(pattern)
        
        # crop to actual raft triangle
        crop_poly = np.array(self.polygon2D)
        crop_path = Path(crop_poly, closed=False)
        included = crop_path.contains_points(pattern)
        pattern = pattern[included, :]
        
        # mirroring cases
        if flip == 'default':
            pass
        elif flip == 'vertical':
            pattern[:, 1] *= -1
        elif flip == 'horizontal':
            pattern[:, 0] *= -1
        else:
            assert False, f'did not recognize flip argument "{flip}"'

        return pattern.tolist()
    
    def circle_intersects(self, x, y, r, resolution=32):
        '''Returns boolean whether a circle of given radius intersects the profile
        polygon. Number of segments used to represent the circle internally can be
        set with arg resolution.'''
        perimeter = self.polygon2D
        perimeter += [perimeter[0]]
        theta = np.linspace(0, np.pi*2, resolution + 1)
        circle_x = x + r*np.cos(theta)
        circle_y = y + r*np.sin(theta)
        collisions = Raft.polygons_collide(perimeter, np.transpose([circle_x, circle_y]))
        return collisions

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    raft = Raft(sphR=4478.677)
    
    # trillium test case
    #trillium_profile = RaftProfile(tri_base=13.821, length=216., chamfer=2.2)
    #raft = Raft(sphR=4478.677, outer_profile=trillium_profile, instr_profile=trillium_profile)

    pattern2D = raft.instr_profile.generate_robot_pattern()
    print('pattern of robot centers (2D):\n', pattern2D)
    print('n robots in pattern', len(pattern2D))
    print('num robots (as calculated by raft)', raft.n_robots)
    robots = raft.generate_robots_table(global_coords=True)
    pattern3D = np.array([robots['x'].data, robots['y'].data, robots['z'].data])
    print('pattern of robot centers (3D):\n', pattern3D)      
    outline = np.transpose(raft.front_poly(instr=True))
    outline_x = outline[0].tolist() + [outline[0, 0]]
    outline_y = outline[1].tolist() + [outline[1, 0]]
    plt.plot(outline_x, outline_y, 'k-')
    plt.plot(robots['x'], robots['y'], 'bo')
    perimeter = robots[robots['intersects perimeter']]
    plt.plot(perimeter['x'], perimeter['y'], 'rx')
    plt.axis('equal')
    plt.show()
    
    pass