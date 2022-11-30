import math
import numpy as np
from scipy.spatial.transform import Rotation  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from astropy import Table

_raft_id_counter = 0

class Raft:
    '''Represents a single triangular raft.'''
    
    def __init__(self, x=0., y=0., spin0=0., focus_offset=0.,
                 outer_profile=None, instr_profile=None,
                 r2nut=None, r2z=None, sphR=math.inf,
                 robot_pitch=6.2, robot_patrol_radius=3.6,
                 ):
        '''
        x ... [mm] x location of center of front triangle
        y ... [mm] y location of center of front triangle
        spin0 ... [deg] rotation of triangle, *not* including precession compensation
        focus_offset ... [mm] focus shift of raft, along z-axis local to the raft
        outer_profile ... RaftProfile instance, defining outer geometry
        instr_profile ... RaftProfile instance, defining instrumented geometry
        r2nut ... function for converting focal plane radius to nutation angle of raft
        r2z ... function for converting focal plane radius to z position of raft
        sphR ... spherical radius which approximates focal surface; sign convention is:
                 sphR > 0 means concave, sphR < 0 means convex, sphR ~ infinity means flat
        robot_pitch ... center-to-center distance between robots within the raft
        robot_patrol_radius ... reach of fi
        '''
        global _raft_id_counter
        self.id = _raft_id_counter
        _raft_id_counter += 1
        self.x = x
        self.y = y
        self.spin0 = spin0
        self.neighbors = []
        self.focus_offset = focus_offset
        self.outer_profile = outer_profile if outer_profile else RaftProfile()
        self.instr_profile = instr_profile if instr_profile else RaftProfile(tri_base=79., chamfer=7.9)
        self.r2nut = r2nut if r2nut else lambda x: np.zeros(np.shape(x))
        self.r2z = r2z if r2z else lambda x: np.zeros(np.shape(x))
        self.sphR = sphR
        self.robot_pitch = robot_pitch

    @property
    def r(self):
        '''radial position [mm] of center of raft at front'''
        return math.hypot(self.x, self.y)
    
    @property
    def z(self):
        '''z position [mm] of center of raft at front'''
        offset_correction = self.focus_offset * math.cos(math.radians(self.nutation))
        return float(self.r2z(self.r)) + offset_correction

    @property
    def precession(self):
        '''angular position [deg] about the z-axis, same as precession'''
        return math.degrees(math.atan2(self.y, self.x))

    @property
    def nutation(self):
        '''angle [deg] w.r.t. z-axis (i.e. matches chief ray at center of raft)'''
        return float(self.r2nut(self.r))
    
    @property
    def spin(self):
        '''rotation [deg] about raft's local z-axis, *including* compensation for
        precession (since raft orientation is defined by a 3-2-3 Euler rotation)'''
        return float(self.spin0 - self.precession)

    def front_poly(self, instr=False):
        '''Nx3 list of polygon vertices giving raft profile at front (i.e. at focal
        surface). Set arg instr=True to use the smaller instrumented area profile'''
        profile = self.instr_profile if instr else self.outer_profile
        poly = profile.polygon3D + [0, 0, self.focus_offset]
        array = self._place_poly(poly)
        return array.tolist()

    @property
    def rear_poly(self):
        '''Nx3 list of polygon vertices giving raft profile at rear (i.e. at
        connectors bulkhead, etc)'''
        poly = np.array(self.outer_profile.polygon3D) + [0, 0, self.focus_offset - self.outer_profile.RL]
        array = self._place_poly(poly)
        return array.tolist()

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
            intersects_perimeter ... whether each robot's patrol disk intersects outline of raft
        '''
        points2D = self.instr_profile.generate_robot_pattern(pitch=self.robot_pitch)
        points2D = np.transpose(points2D)
        local_r = np.hypot(points2D[0], points2D[1])
        if np.isneginf(self.sphR) or np.isposinf(self.sphR):
            dz = np.zeros(np.shape(local_r))
        else:
            dz = self.sphR - (self.sphR**2 - local_r**2)**0.5
        points3D = np.transpose(np.append(points2D, [dz], axis=0))
        if global_coords:
            points3D = self._place_poly(points3D)
            angles = np.ones_like(points3D) * [self.precession, self.nutation, self.spin]
        else:
            angles = np.zeros_like(points3D)
        data = {'idx': np.arange(len(points3D[0])),
                'x': points3D[0],
                'y': points3D[1],
                'z': points3D[2],
                'precession': angles[0],
                'nutation': angles[1],
                'spin': angles[2],
                'intersects_perimeter': ,
                }
        table = Table(data)
        return table
    
    @property
    def robot_angles(self):
        '''Nx3 list of (precession, nutation, spin) coordinates of the individual
        robots on the raft. These are in the global coordinate system of the focal
        plane.'''
        # 2022-11-30 - JHS - Current design assumption is we will keep all robot
        # center axes parallel to the raft.
        points3D = self.robot_centers
        
        return angles

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

    def _place_poly(self, poly):
        '''Transform a polygon (N x 3) from the origin to the raft's center position on the
        focal surface. The polygon is first rotated such that a vector (0, 0, 1) becomes
        its final orientation when placed at the corresponding radius, and such that a point
        (0, 0, 0) will land on the focal surface.'''
        rot = Rotation.from_euler('ZYZ', (self.precession, self.nutation, self.spin), degrees=True)
        rotated = rot.apply(poly)
        translated = rotated + [self.x, self.y, self.r2z(self.r)]
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

    def generate_robot_pattern(self, pitch=6.2):
        '''Produce 2D pattern of robots that fit within the polygon.
          INPUTS:   pitch ... [mm] center-to-center distance between robots within the raft
          OUTPUTS:  Nx2 list of (x, y) positions
        '''
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
        pattern = pattern[:, included]

        return pattern.tolist()
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    raft = Raft(sphR=4000)
    pattern2D = raft.instr_profile.generate_robot_pattern()
    print('pattern of robot centers (2D):\n', pattern2D)
    print('n_robots_in_pattern', len(pattern2D[0]))
    pattern3D = raft.robot_centers
    print('pattern of robot centers (3D):\n', pattern3D)      
    outline = np.transpose(raft.front_poly(instr=True))
    outline_x = outline[0].tolist() + [outline[0, 0]]
    outline_y = outline[1].tolist() + [outline[1, 0]]
    plt.plot(outline_x, outline_y, 'k-')
    plt.plot(np.transpose(pattern2D)[0], np.transpose(pattern2D)[1], 'bo')
    plt.axis('equal')
    plt.show()
    pass