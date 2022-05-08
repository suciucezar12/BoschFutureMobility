import math

import cv2
import numpy as np


class Tracker:

    def __init__(self, x_car=None, y_car=None, v=None, dt=None, theta_yaw_map=None, yaw=None, x_end=None, y_end=None,
                 final_thresh=None, ref_thresh=None, pixel_resolution=None, points_ref=None,
                 size=None):
        # info car
        self.x_car = x_car
        self.y_car = y_car
        self.theta_car = theta_yaw_map  # heading of car
        self.v = v  # baseline speed
        self.dt = dt

        self.theta_offset = (theta_yaw_map - self.yaw_to_trigo(yaw) + 360) % 360

        self.x_end = x_end
        self.y_end = y_end

        self.final_thresh = final_thresh
        self.ref_thresh = ref_thresh
        self.pixel_resolution = pixel_resolution

        self.points_ref = points_ref

        self.size = size
        self.map = np.zeros((self.size, self.size), dtype="unint8")

        for point_ref in self.points_ref:
            x, y = point_ref
            cv2.circle(self.map, (x, abs(y - self.size)), 2, (0, 0, 255), 2)


    def yaw_to_trigo(self, yaw):
        return 360 - (yaw + 270) % 360

    def distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * self.pixel_resolution

    def line_intersection(self, eq1, eq2):
        m1, b1 = eq1
        m2, b2 = eq2
        x = (b1 - b2) / (m2 - m1)
        y = m1 * x + b1
        return x, y

    def get_ref_point(self):
        point_ref_min = None
        min_dist = 100000000

        for point_ref in self.points_ref:
            x_ref, y_ref = point_ref
            slope_car = math.tan(math.radians(self.theta_car))
            eq_car = [slope_car, self.y_car - slope_car * self.x_car]

            slope_perp_car = math.tan(math.radians((slope_car + 90) // 360))
            eq_perp_car = [slope_perp_car, self.y_car - slope_perp_car * self.x_car]

            eq_ref = [slope_car, y_ref - slope_car * x_ref]

            x_int, y_int = self.line_intersection(eq_perp_car, eq_ref)

            if y_ref - y_int >= 0:
                d = self.distance((x_ref, y_ref), (self.x_car, self.y_car))
                if d >= self.ref_thresh and d >= min_dist:
                    point_ref_min = point_ref
                    min_dist = d

        return point_ref_min

    def generate_circle_points(self, r=None, d=None, x_c=None, y_c=None, alpha_min=None, alpha_max=None):
        points = []
        alpha = 2 * math.asin(float(d / (2 * r)))
        k = 0
        while k * alpha <= alpha_max:
            x = r * math.cos(k * alpha)
            y = r * math.sin(k * alpha)
            points.append((x + x_c, y + y_c))
        return points

    def run(self):
        # DO WHILE
        while self.distance((self.x_car, self.y_car), (self.x_end, self.y_end)) >= self.final_thresh:
            # yaw = 0  # data from brain
            # yaw = (self.yaw_to_trigo(yaw) - self.theta_offset + 360) % 360
            # self.theta_car = yaw
            #
            # self.x_car = self.x_car + self.v * math.cos(math.radians(self.theta_car)) * self.dt
            # self.y_car = self.y_car + self.v * math.sin(math.radians(self.theta_car)) * self.dt
            # cv2.circle(self.map, (self.x_car, abs(self.y_car - self.size)), 2, (255, 255, 255), 2)

            point_ref = self.get_ref_point()
            x_ref, y_ref = point_ref
            cv2.line(self.map, (self.x_car, abs(self.y_car - self.size)), (x_ref, abs(y_ref - self.size)), (0, 255, 0), 1)

            theta_ref = (math.atan((y_ref - self.y_car) / (x_ref - self.x_car)) + 360) // 360

            steering_angle = self.theta_car - theta_ref  # data goes to the brain

            yaw = 0  # data from brain
            yaw = (self.yaw_to_trigo(yaw) - self.theta_offset + 360) % 360
            self.theta_car = yaw

            self.x_car = self.x_car + self.v * math.cos(math.radians(self.theta_car)) * self.dt
            self.y_car = self.y_car + self.v * math.sin(math.radians(self.theta_car)) * self.dt
            cv2.circle(self.map, (self.x_car, abs(self.y_car - self.size)), 2, (255, 255, 255), 2)


class Map():

    def __init__(self, size=None, pixel_resolution=None):
        self.size = size
        self.map = np.zeros((self.size, self.size), np.uint8)
        self.pixel_resolution = pixel_resolution
        # self.points_ref = self.generate_circle_points(r=int(90 / self.pixel_resolution), d=int(9 / self.pixel_resolution), x_c=int(30 / self.pixel_resolution), y_c=abs(int(30 / self.pixel_resolution) - self.size), alpha_min=0, alpha_max=1.57)
        self.points_ref = []
        self.points_ref = self.generate_circle_points(r=90,
                                                      d=9,
                                                      x_c=30,
                                                      y_c=30, alpha_min=0,
                                                      alpha_max=1.57)
        self.init_map()
        self.points_ref = self.generate_circle_points(r=60,
                                                      d=9,
                                                      x_c=180,
                                                      y_c=30, alpha_min=1.57,
                                                      alpha_max=3.14)
        self.init_map()
        self.points_ref = self.generate_line_points(x1=120, y1=30, x2=120, y2=180, n=10)
        self.init_map()

        # for point in self.points_ref:
        #     x, y = point
        #     print("x = {}, y = {}".format(int(x), int(abs(y - self.size))))
        #     cv2.circle(self.map, (int(x), int(abs(y - self.size))), 2, (0, 0, 255), 2)

    def init_map(self):
        for point in self.points_ref:
            x, y = point
            # print("x = {}, y = {}".format(int(x), int(abs(y - self.size))))
            # cv2.circle(self.map, (int(x / self.pixel_resolution), int(abs(y / self.pixel_resolution - self.size))), 2, (0, 0, 255), 2)
            # self.map[int(x / self.pixel_resolution)][int(abs(y / self.pixel_resolution - self.size))] = 255
            self.map[int(abs(y / self.pixel_resolution - self.size))][int(x / self.pixel_resolution)] = 255
            # print(self.map)
            cv2.imshow("map", self.map)
            cv2.waitKey(0)

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

map = Map(size=500, pixel_resolution=float(210 / 500))
# map.init_map()
cv2.imshow("Map", map.map)
cv2.waitKey(0)
