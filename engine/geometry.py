import cv2
import numpy as np
from shapely.geometry import Point

class GeometryEngine:
    def __init__(self):
        # Standard BWF Singles court: 13.4m x 5.18m
        # We map to singles lines as user selects the inner lines.
        # Real world coordinates: (0,0) is top-left, (5.18, 13.4) is bottom-right.
        self.real_corners = np.float32([[0,0], [5.18,0], [0,13.4], [5.18,13.4]])
        self.matrix = None
        self.inv_matrix = None

    def calculate_homography(self, pixel_corners):
        """
        Calculates the homography matrix from pixel corners to real-world corners.
        pixel_corners: List of 4 (x, y) tuples. Order: TL, TR, BL, BR.
        """
        src_points = np.float32(pixel_corners)
        self.matrix = cv2.getPerspectiveTransform(src_points, self.real_corners)
        # Calculate inverse matrix (Meters -> Pixels)
        _, self.inv_matrix = cv2.invert(self.matrix)
        return self.matrix

    def transform_point(self, point):
        """
        Transforms a single point (x, y) from pixel space to meter space.
        Optimization: Uses manual calculation to avoid cv2.perspectiveTransform overhead for single points.
        """
        if self.matrix is None:
            return None
        
        # Optimization: Manual Homography Transform (3.3x faster for single points)
        # x' = (m00*x + m01*y + m02) / (m20*x + m21*y + m22)
        # y' = (m10*x + m11*y + m12) / (m20*x + m21*y + m22)
        
        x, y = point
        m = self.matrix

        # Denominator (w')
        div = m[2, 0] * x + m[2, 1] * y + m[2, 2]

        if div == 0:
            return None

        inv_div = 1.0 / div

        tx = (m[0, 0] * x + m[0, 1] * y + m[0, 2]) * inv_div
        ty = (m[1, 0] * x + m[1, 1] * y + m[1, 2]) * inv_div

        return np.array([tx, ty], dtype=np.float32)

    def get_relative_coordinates(self, point_m):
        """
        Converts Absolute Coordinates (0-6.1, 0-13.4) to 
        Center-Net Relative Coordinates (CoachAI Standard).
        
        Center: (3.05, 6.7) -> (0, 0)
        X: Positive Right, Negative Left
        Y: Positive Bottom (Far), Negative Top (Near)
        """
        if point_m is None: return None
        x_abs, y_abs = point_m
        
        # Center of court
        CENTER_X = 3.05
        CENTER_Y = 6.7
        
        return (x_abs - CENTER_X, y_abs - CENTER_Y)

    def calculate_dynamic_base(self, shot_type, player_pos):
        """
        Calculates the ideal base position based on the shot played.
        Center Base is roughly (2.59, 6.7) for Singles.
        """
        center_x, center_y = 2.59, 6.7
        
        if shot_type == 'net_drop':
             # If I hit to net, base is closer to front to cover return net shot
             return (center_x, center_y - 1.0) 
        elif shot_type == 'smash':
            # Identify recovery after smash (usually stay slightly back or center)
             return (center_x, center_y)
             
        return (center_x, center_y)

    def is_in_play_area(self, point_px, polygon_px):
        """
        Checks if a point is within the defining polygon (Court + Margin).
        Uses OpenCV's pointPolygonTest.
        
        point_px: (x, y) tuple in pixels
        polygon_px: List of (x, y) tuples representing the polygon OR numpy array (N, 1, 2)
        """
        # cv2.pointPolygonTest requires contour to be float32/int32 array (N, 1, 2)
        if isinstance(polygon_px, np.ndarray):
            contour = polygon_px
        else:
            contour = np.array(polygon_px, dtype=np.int32).reshape((-1, 1, 2))
        
        # Measure distance. >= 0 means inside or on edge.
        result = cv2.pointPolygonTest(contour, point_px, False)
        return result >= 0

    def get_play_area_polygon(self, margin=2.0):
        """
        Returns the pixel polygon of the court expanded by 'margin' meters.
        """
        if self.inv_matrix is None:
            return []
            
        # Define extended corners in meters
        # Real corners: (0,0), (5.18,0), (0,13.4), (5.18,13.4) -- Wait, self.real_corners order?
        # self.real_corners definition: [[0,0], [5.18,0], [0,13.4], [5.18,13.4]] (TL, TR, BL, BR)
        
        # Extended box:
        x_min, x_max = 0 - margin, 5.18 + margin
        y_min, y_max = 0 - margin, 13.4 + margin
        
        # Order: TL, TR, BR, BL (for proper polygon drawing)
        pts_m = np.float32([
            [x_min, y_min], # TL
            [x_max, y_min], # TR
            [x_max, y_max], # BR
            [x_min, y_max]  # BL
        ]).reshape(-1, 1, 2)
        
        # Transform to pixels
        pts_px = cv2.perspectiveTransform(pts_m, self.inv_matrix)
        
        # Reshape to list of tuples
        return [tuple(p[0]) for p in pts_px]

class ZoneMapper:
    """
    Maps court coordinates to tactical zones (CoachAI style).
    Zones:
    - 0-3: Depth (0=Net, 1=Front, 2=Mid, 3=Back)
    - A-C: Width (A=Left, B=Center, C=Right)
    """
    def __init__(self):
        # Meters from net (Total length 13.4m, Net at 6.7m)
        # We work in "Distance from Net" absolute
        self.NET_M = 0.0
        self.FRONT_LINE_M = 1.98 # Short Service Line
        self.MID_LINE_M = 3.96   # Approx mid court (Doubles service line is at back-ish)
        self.BACK_LINE_M = 6.7   # Base line
        
        # Width (Total 6.1m, Center 3.05m)
        self.CENTER_M = 3.05
        self.SIDE_LINE_M = 0.0 # and 6.1
        self.SINGLES_LINE_M = 0.46
        
    def get_zone(self, pos_m):
        """
        pos_m: (x, y) in meters. 
        Assumes court coordinate system:
        X: 0 to 6.1 (Left to Right)
        Y: 0 to 13.4 (Top to Bottom). Net at 6.7.
        """
        if pos_m is None: return "Unknown"
        
        x, y = pos_m
        
        # Normalize to "Distance from Net" and "Side"
        dist_from_net = abs(y - 6.7)
        side = "Top" if y < 6.7 else "Bottom"
        
        # Depth Code (0, 1, 2, 3)
        if dist_from_net < 0.5:
             depth = "0" # Net Play
        elif dist_from_net < 2.5:
             depth = "1" # Front
        elif dist_from_net < 4.5:
             depth = "2" # Mid
        else:
             depth = "3" # Rear
             
        # Width Code (A, B, C)
        # Center width ~2m?
        # A=Left, B=Center, C=Right
        if x < 2.0:
            width = "A"
        elif x > 4.1:
            width = "C"
        else:
            width = "B"
            
        return f"{width}{depth}" # e.g. "C3" (Right Rear)
