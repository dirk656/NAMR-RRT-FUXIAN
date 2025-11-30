import numpy as np


def det(a, b):
    """计算两个二维向量的叉积，用于判断线段是否相交"""
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(line1, line2):
    """
    判断两条线段是否相交
    
    参数:
        line1: [[x_start, y_start], [x_end, y_end]]，第一条线段
        line2: [[x_start, y_start], [x_end, y_end]]，第二条线段
    
    返回:
        bool: 如果两条线段相交则返回True，否则返回False
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    
    # 如果叉积为0，表示两条线段平行或共线，不相交
    if div == 0:
        return False
    
    # 计算两条线段的交点坐标
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    # 使用数值容差eps来处理浮点数精度问题
    eps = 1e-6
    # 检查交点是否同时位于两条线段内部
    if (min(line1[0][0], line1[1][0]) - eps <= x <= max(line1[0][0], line1[1][0]) + eps and
        min(line1[0][1], line1[1][1]) - eps <= y <= max(line1[0][1], line1[1][1]) + eps and
        min(line2[0][0], line2[1][0]) - eps <= x <= max(line2[0][0], line2[1][0]) + eps and
        min(line2[0][1], line2[1][1]) - eps <= y <= max(line2[0][1], line2[1][1]) + eps):
        return True
    return False


def check_collision_line_single_circle(line, circle_center, circle_radius, clearance=0):
    """
    检查一条线段是否与单个圆形障碍物发生碰撞
    
    参数:
        line: [[x_start, y_start], [x_end, y_end]]，待检测的线段
        circle_center: [x_center, y_center]，圆心坐标
        circle_radius: 圆的半径
        clearance: 安全余量
    
    返回:
        bool: 如果线段与圆发生碰撞（考虑安全余量）则返回True
    """
    circle_radius_with_clearance = circle_radius + clearance  # 考虑安全余量的有效半径
    line_vector = line[1] - line[0]
    line_length = np.linalg.norm(line_vector)
    
    # 如果线段长度为0，只检查起点是否在圆内
    if line_length == 0:
        return point_in_single_circle(line[0], circle_center, circle_radius, clearance)
    
    line_direction = line_vector / line_length  # 线段方向向量
    start_to_center = circle_center - line[0]  # 起点到圆心的向量
    
    # 计算线段上离圆心最近点的投影
    projection = np.dot(start_to_center, line_direction)
    closest_point = np.clip(projection, 0, line_length) * line_direction + line[0]
    
    # 计算最近点到圆心的距离
    distance = np.linalg.norm(np.array(circle_center) - closest_point)
    
    # 如果最近点到圆心的距离小于有效半径，则发生碰撞
    return distance <= circle_radius_with_clearance


def point_in_single_circle(point, circle_center, circle_radius, clearance=0):
    """检查单个点是否在圆形障碍物内部（考虑安全余量）"""
    return np.linalg.norm(point - circle_center) <= circle_radius + clearance


def point_in_single_rectangle(point, xywh, clearance=0):
    """检查单个点是否在矩形障碍物内部（考虑安全余量）"""
    x, y, w, h = xywh
    # 检查点是否在扩展后的矩形范围内（考虑安全余量）
    return (x - clearance <= point[0] <= x + w + clearance and 
            y - clearance <= point[1] <= y + h + clearance)


def check_collision_line_single_rectangle(line, xywh, clearance=0):
    """检查一条线段是否与单个矩形障碍物发生碰撞"""
    # 如果线段的任一端点在矩形内，则发生碰撞
    if point_in_single_rectangle(line[0], xywh, clearance) or \
       point_in_single_rectangle(line[1], xywh, clearance):
        return True
    
    # 构造扩展后的矩形四个顶点
    x, y, w, h = xywh
    rect_points = np.array([
        [x - clearance, y - clearance],
        [x + w + clearance, y - clearance],
        [x + w + clearance, y + h + clearance],
        [x - clearance, y + h + clearance],
    ])
    
    # 构造矩形的四条边
    rect_lines = np.array([
        [rect_points[0], rect_points[1]],
        [rect_points[1], rect_points[2]],
        [rect_points[2], rect_points[3]],
        [rect_points[3], rect_points[0]],
    ])
    
    # 检查线段是否与矩形的任一边相交
    for rect_line in rect_lines:
        if line_intersection(line, rect_line):
            return True
    return False


def check_collision_single_aabb_pair(aabb1, aabb2):
    """检查两个轴对齐包围盒（AABB）是否相交"""
    # 检查两个包围盒在x和y方向上是否存在重叠
    return ((aabb1[0][0] <= aabb2[1][0] and aabb1[1][0] >= aabb2[0][0]) and
            (aabb1[0][1] <= aabb2[1][1] and aabb1[1][1] >= aabb2[0][1]))


def check_collsion_aabb_aabbs(aabb, aabbs):
    """批量检查一个包围盒与多个包围盒是否相交"""
    # 使用向量化计算检查包围盒在x和y方向上的重叠关系
    collision = ((aabb[0, 0] <= aabbs[:, 1, 0]) * (aabb[1, 0] >= aabbs[:, 0, 0]) *
                 (aabb[0, 1] <= aabbs[:, 1, 1]) * (aabb[1, 1] >= aabbs[:, 0, 1]))
    return collision


def check_collision_line_circles_rectangles(line, circles, rectangles, clearance=0):
    """
    检查一条线段是否与多个圆形障碍物或矩形障碍物发生碰撞
    
    采用两阶段检测策略：先用包围盒快速筛选，再进行精确碰撞检测
    """
    # 计算线段的包围盒
    line_aabb = np.array([[min(line[0, 0], line[1, 0]), min(line[0, 1], line[1, 1])],
                          [max(line[0, 0], line[1, 0]), max(line[0, 1], line[1, 1])]])
    
    # 检查与圆形障碍物的碰撞
    if circles is not None:
        # 为每个圆形障碍物构造包围盒
        circles_x1 = circles[:, 0] - circles[:, 2] - clearance
        circles_y1 = circles[:, 1] - circles[:, 2] - clearance
        circles_x2 = circles[:, 0] + circles[:, 2] + clearance
        circles_y2 = circles[:, 1] + circles[:, 2] + clearance
        circle_aabbs = np.array([[circles_x1, circles_y1], [circles_x2, circles_y2]]).transpose(2, 0, 1)
        
        # 使用包围盒快速筛选可能发生碰撞的圆形障碍物
        circle_aabb_collisions = check_collsion_aabb_aabbs(line_aabb, circle_aabbs)
        circles_to_check = circles[np.where(circle_aabb_collisions)]
        
        # 对筛选出的圆形障碍物进行精确碰撞检测
        if len(circles_to_check) > 0:
            for circle_to_check in circles_to_check:
                circle_center, circle_radius = circle_to_check[:2], circle_to_check[2]
                if check_collision_line_single_circle(line, circle_center, circle_radius, clearance):
                    return True
    
    # 检查与矩形障碍物的碰撞
    if rectangles is not None:
        # 为每个矩形障碍物构造包围盒
        rectangles_x1 = rectangles[:, 0] - clearance
        rectangles_y1 = rectangles[:, 1] - clearance
        rectangles_x2 = rectangles[:, 0] + rectangles[:, 2] + clearance
        rectangles_y2 = rectangles[:, 1] + rectangles[:, 3] + clearance
        rectangle_aabbs = np.array([[rectangles_x1, rectangles_y1], [rectangles_x2, rectangles_y2]]).transpose