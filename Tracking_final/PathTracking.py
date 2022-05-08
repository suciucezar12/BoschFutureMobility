import numpy as np
import math


class PathGenerator:

    def generate_circle_points(self, r=None, d=None, x_c=None, y_c=None, alpha_min=None, alpha_max=None):
        points = []
        alpha = 2 * math.asin(float(d / (2 * r)))
        k = 0
        while k * alpha + alpha_min <= alpha_max:
            x = r * math.cos((k * alpha + alpha_min) % alpha_max)
            y = r * math.sin((k * alpha + alpha_min) % alpha_max)
            points.append((x + x_c, y + y_c))
            k += 1
        return points

    def generate_line_points(self, x1=None, y1=None, x2=None, y2=None, n=None):
        if x1 != x2:
            x_points = np.linspace(x1, x2, num=n, endpoint=True)
        else:
            x_points = np.zeros(n) + x1
        if y1 != y2:
            y_points = np.linspace(y1, y2, num=n, endpoint=True)
        else:
            y_points = np.zeros(n) + y1

        points = []
        for x, y in zip(x_points, y_points):
            points.append((x, y))
        return points


class Map:

    def __init__(self, size_pixel=None, size_cm=None, ref_points=None):
        self.size_pixel = size_pixel
        self.pixel_resolution = float(size_cm / size_pixel)
        self.map = np.zeros((self.size_pixel, self.size_pixel), np.uint8)

        # draw ref points
        for point in ref_points:
            x, y = point
            x, y = self.get_point_map(x, y)
            self.map[int(abs(y / self.pixel_resolution - self.size_pixel))][int(x / self.pixel_resolution)] = 255

    def get_point_map(self, x, y):
        # self.map[int(abs(y / self.pixel_resolution - self.size_pixel))][int(x / self.pixel_resolution)] = 255
        return int(x / self.pixel_resolution), int(abs(y / self.pixel_resolution - self.size_pixel))


class PathTracking:

    def __init__(self, map=None, ref_points=None, size_pixel=None, size_cm=None,
                 x_car=None, y_car=None, theta_yaw_map=None, yaw=None, v=None, dt=None,
                 ref_thresh=None, final_thresh=None, end_point=None,
                 ):
        # info about the map
        self.map = map(size_pixel=size_pixel, size_cm=size_cm, ref_points=ref_points)
        self.ref_points = ref_points

        # info about the car
        self.x_car = x_car
        self.y_car = y_car
        self.theta_car = theta_yaw_map  # heading angle of the car wrt ot the map cs
        self.theta_offset = (theta_yaw_map - self.yaw_to_trigo(yaw) + 360) % 360
        self.v = v
        self.dt = dt

        # destination info
        self.x_end, self.y_end = end_point
        self.ref_thresh = ref_thresh
        self.final_thresh = final_thresh

    def get_ref_point(self):
        point_ref_min = None
        min_dist = 100000000

        for point_ref in self.ref_points:
            x_ref, y_ref = point_ref
            slope_car = math.tan(math.radians(self.theta_car))
            slope_perp_car = math.tan(math.radians((self.theta_car + 90) // 360))

            eq_perp_car = self.get_line_eq(slope_perp_car, (self.x_car, self.y_car))
            eq_ref = self.get_line_eq(slope_car, (x_ref, y_ref))

            x_int, y_int = self.line_intersection(eq_perp_car, eq_ref)

            if y_ref - y_int >= 0:
                d = self.distance((x_ref, y_ref), (self.x_car, self.y_car))
                if d >= self.ref_thresh and d >= min_dist:
                    point_ref_min = point_ref
                    min_dist = d

        return point_ref_min

    def yaw_to_trigo(self, yaw):
        return 360 - (yaw + 270) % 360

    def distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def line_intersection(self, eq1, eq2):
        # y = m*x + c
        m1, c1 = eq1
        m2, c2 = eq2
        x = (c1 - c2) / (m2 - m1)
        y = m1 * x + c1
        return x, y

    def get_line_eq(self, slope, point):
        x, y = point
        c = y - slope * x
        return (slope, c)

    def run(self):
        while self.distance((self.x_car, self.y_car), (self.x_end, self.y_end)) >= self.final_thresh:

            point_ref = self.get_ref_point()
            x_ref, y_ref = point_ref

            theta_ref = (math.atan((y_ref - self.y_car) / (x_ref - self.x_car)) + 360) // 360

            steering_angle = self.theta_car - theta_ref  # data goes to the brain

            yaw = 0  # yaw data from IMU
            yaw = (self.yaw_to_trigo(yaw) - self.theta_offset + 360) % 360
            self.theta_car = yaw

            self.x_car = self.x_car + self.v * math.cos(math.radians(self.theta_car)) * self.dt
            self.y_car = self.y_car + self.v * math.sin(math.radians(self.theta_car)) * self.dt
